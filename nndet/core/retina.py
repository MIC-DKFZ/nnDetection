# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from nndet.core import boxes as box_utils
from nndet.core.abstract import AbstractDetector
from nndet.core.boxes.anchors import AnchorGeneratorType
from nndet.core.boxes.assign import assign_targets_to_anchors
from nndet.core.post.box import BoxPostprocessing
from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.heads.comb import AnchorHeadType
from nndet.nn.heads.segmenter import SegmenterType
from nndet.nn.neck.abstract import AbstractNeck


class BaseRetinaNet(AbstractDetector):
    def __init__(
        self,
        dim: int,
        # modules
        backbone: AbstractBackbone,
        neck: AbstractNeck,
        head: AnchorHeadType,
        anchor_generator: AnchorGeneratorType,
        matcher: box_utils.MatcherType,
        box_post: BoxPostprocessing,
        decoder_levels: tuple = (2, 3, 4, 5),
        segmenter: Optional[SegmenterType] = None,
    ):
        """
        Base Retina(U)Net
        Can be subclasses to add specific configurations to it

        Args:
            dim: number of spatial dimensions
            backbone: encoder module
            neck: decoder module
            head: head module
            anchor_generator: generate anchors
            matcher: match ground truth boxes and anchors
            box_post: module responsible to postprocess the boxes (clipping,
                nms, ...) and generate the class labels
            decoder_levels: decoder levels to use for detection prediciton
            segmenter: segmentation module
        """
        super().__init__()
        assert dim in [2, 3]
        self.dim = dim
        self.decoder_levels = decoder_levels

        self.backbone = backbone
        self.neck = neck
        self.head = head

        self.anchor_generator = anchor_generator
        self.proposal_matcher = matcher
        self.box_post = box_post

        self.segmenter = segmenter

    def forward(
        self,
        inp: torch.Tensor,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        List[torch.Tensor],
        Dict[str, torch.Tensor],
        List[torch.Tensor],
    ]:
        """
        Compute predicted bounding boxes, scores and segmentations

        Args:
            inp (torch.Tensor): batch of input images

        Returns:
            dict: predictions from head. Typically includes

                ``"box_deltas"`` Tensor
                    bounding box offsets [Num_Anchors_Batch, (dim * 2)]

                ``"box_logits"`` Tensor
                    classification logits  [Num_Anchors_Batch, (num_classes)]

            List[torch.Tensor]: list of anchors (for each image inside the
                batch)
            dict: segmentation prediction. None if retina net is configured.
                Typically includes

                ``"seg_logits"`` Tensor
                    segmentation logits

            List[torch.Tensor]: feature maps from decoder
        """
        features_maps_all = self.neck(self.backbone(inp))
        feature_maps_head = [features_maps_all[i] for i in self.decoder_levels]

        pred_detection = self.head(feature_maps_head)
        anchors = self.anchor_generator(inp, feature_maps_head)

        pred_seg = (
            self.segmenter(features_maps_all) if self.segmenter is not None else None
        )
        return pred_detection, anchors, pred_seg, features_maps_all

    def train_step(
        self,
        images: Tensor,
        targets: dict,
        batch_num: int,
    ) -> Dict[str, torch.Tensor]:
        """
        See `self.train_step_with_features` for more info
        """
        losses, _, _ = self.train_step_with_features(
            images=images,
            targets=targets,
            predict=False,
            batch_num=batch_num,
        )
        return losses

    @torch.no_grad()
    def validation_step(
        self,
        images: Tensor,
        targets: dict,
        batch_num: bool,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        See `self.train_step_with_features` for more info
        """
        losses, prediction, _ = self.train_step_with_features(
            images=images,
            targets=targets,
            predict=True,
            batch_num=batch_num,
        )
        return losses, prediction

    @torch.no_grad()
    def inference_step(
        self,
        images: Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        See `inference_step_with_features` for more info
        """
        prediction, _ = self.inference_step_with_features(images=images, **kwargs)
        return prediction

    def train_step_with_features(
        self,
        images: Tensor,
        targets: dict,
        predict: bool,
        batch_num: int,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict], List[torch.Tensor]]:
        """
        Perform a single training step (forward pass + loss computation)

        Args:
            images: batch of images
            targets: labels for training

                ``"target_boxes"`` (List[Tensor])
                    ground truth bounding boxes  (x1, y1, x2, y2, (z1, z2))
                    [X, dim * 2], X= number of  ground truth boxes in image

                ``"target_classes"`` (List[Tensor])
                    ground truth class per box (classes start from 0) [X],
                    X= number of ground truth boxes in image

                ``"target_seg"`` (Tensor)
                    segmentation ground truth (only needed if ::param::`segmenter`
                    was provided in init) (classes start from 1, 0 background)

            predict: compute final predictions (includes detection
                postprocessing)
            batch_num: batch index inside epoch

        Returns:
            Dict: all losses
            Dict: predictions for metric calculation

                ``"pred_boxes"`` List[Tensor]
                    predicted bounding boxes for each image List[[R, dim * 2]]

                ``"pred_scores"`` List[Tensor]
                    predicted probability for the class List[[R]]

                ``"pred_labels"`` List[Tensor]
                    predicted class List[[R]]

                ``"pred_seg"`` Tensor
                    predicted segmentation [N, dims]

            List[torch.Tensor]: feature maps from decoder
        """
        target_boxes: List[Tensor] = targets["target_boxes"]
        target_classes: List[Tensor] = targets["target_classes"]
        target_seg: Tensor = targets.get("target_seg", None)

        pred_detection, anchors, pred_seg, features = self(images)

        labels, matched_gt_boxes, _ = assign_targets_to_anchors(
            proposal_matcher=self.proposal_matcher,
            anchors=anchors,
            target_boxes=target_boxes,
            target_classes=target_classes,
            num_anchors_per_level=self.anchor_generator.get_num_acnhors_per_level(),
            num_anchors_per_loc=self.anchor_generator.num_anchors_per_location()[0],
        )

        losses = {}
        head_losses, pos_idx, neg_idx = self.head.compute_loss(
            pred_detection, labels, matched_gt_boxes, anchors
        )
        losses.update(head_losses)

        if self.segmenter is not None:
            assert target_seg is not None, "FIXME"  # FIXME: better handling here
            losses.update(self.segmenter.compute_loss(pred_seg, target_seg))

        if predict:
            prediction = self.postprocess_for_inference(
                images=images,
                pred_detection=pred_detection,
                pred_seg=pred_seg,
                anchors=anchors,
            )
        else:
            prediction = None
        return losses, prediction, features

    def inference_step_with_features(
        self,
        images: Tensor,
        **kwargs,
    ) -> Union[Dict[str, Any], List[torch.Tensor]]:
        """
        Perform inference for a batch of images

        Args:
            images: batch of input images [N, C, W, H, (D)]

        Returns:
            Dict: predictions

                ``"pred_boxes"`` List[Tensor]
                    predicted bounding boxes for each image List[[R, dim * 2]]

                ``"pred_scores"`` List[Tensor]
                    predicted probability for the class List[[R]]

                ``"pred_labels"`` List[Tensor]
                    predicted class List[[R]]

                ``"pred_seg"`` Tensor
                    predicted segmentation [N, C, dims]

            List[torch.Tensor]: feature maps from encoder
        """
        pred_detection, anchors, pred_seg, features = self(images)
        prediction = self.postprocess_for_inference(
            images=images,
            pred_detection=pred_detection,
            anchors=anchors,
            pred_seg=pred_seg,
        )
        return prediction, features

    @torch.no_grad()
    def postprocess_for_inference(
        self,
        images: torch.Tensor,
        pred_detection: Dict[str, torch.Tensor],
        anchors: List[torch.Tensor],
        pred_seg: Dict[str, torch.Tensor],
    ) -> Dict[str, Union[List[Tensor], Tensor]]:
        """
        Postprocess predictions for inference

        Args:
            images: input images
            pred_detection: detection predictions
            anchors: anchors
            pred_seg: segmentation predictions

        Returns:
            Dict: post processed predictions

                ``"pred_boxes"`` List[Tensor]
                    predicted bounding boxes for each image List[[R, dim * 2]]

                ``"pred_scores"`` List[Tensor]
                    predicted probability for the class List[[R]]

                ``"pred_labels"`` List[Tensor]
                    predicted class List[[R]]

                ``"pred_seg"`` Tensor
                    predicted segmentation [N, C, dims]

        """
        image_shapes = [images.shape[2:]] * images.shape[0]
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in anchors]

        # handle boxes (e.g. apply deltas when L1 loss is used), convert logits into probs
        pred_detection = self.head.postprocess_for_inference(pred_detection, anchors)
        pred_boxes, pred_probs = (
            pred_detection["pred_boxes"],
            pred_detection["pred_probs"],
        )

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_probs = pred_probs.split(boxes_per_image, 0)

        # postprocess predictions -> topk, nms etc. + label creation
        boxes, probs, labels = self.box_post.process_batch(
            reps=pred_boxes,
            probs=pred_probs,
            image_shapes=image_shapes,
        )

        prediction = {"pred_boxes": boxes, "pred_scores": probs, "pred_labels": labels}
        if self.segmenter is not None:
            prediction["pred_seg"] = self.segmenter.postprocess_for_inference(pred_seg)[
                "pred_seg"
            ]
        return prediction
