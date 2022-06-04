# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from torchvision (https://github.com/pytorch/vision) licensed under
# SPDX-FileCopyrightText: 2016 Soumith Chintala
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from nndet.core import boxes as box_utils
from nndet.core.abstract import AbstractDetector
from nndet.core.boxes.anchors import AnchorGeneratorType
from nndet.core.boxes.assign import assign_targets_to_anchors
from nndet.core.boxes.post import post_image_single_class_regression
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
        num_classes: int,
        anchor_generator: AnchorGeneratorType,
        matcher: box_utils.MatcherType,
        decoder_levels: tuple = (2, 3, 4, 5),
        # post-processing
        score_thresh: float = None,
        detections_per_img: int = 100,
        topk_candidates: int = 10000,
        remove_small_boxes: float = 1e-2,
        nms_thresh: float = 0.9,
        # optional
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
            num_classes: number of foreground classes
            anchor_generator: generate anchors
            matcher: match ground truth boxes and anchors
            decoder_levels: decoder levels to use for detection prediciton
            score_thresh: minimum output probability
            detections_per_img: max detections per image
            topk_candidates: select only topk candidates for nms computation
            remove_small_boxes: remove small bounding boxes
            nms_thresh: non maximum suppression threshold
            segmenter: segmentation module
        """
        super().__init__()
        assert dim in [2, 3]
        self.dim = dim
        self.decoder_levels = decoder_levels

        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.num_foreground_classes = num_classes

        self.anchor_generator = anchor_generator
        self.proposal_matcher = matcher

        self.score_thresh = score_thresh
        self.topk_candidates = topk_candidates
        self.detections_per_img = detections_per_img
        self.remove_small_boxes = remove_small_boxes
        self.nms_thresh = nms_thresh

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
        # import napari
        # with napari.gui_qt():
        #     viewer = napari.view_image(images.detach().cpu().numpy())
        #     viewer.add_labels(seg_targets[:, None].detach().cpu().numpy())

        target_boxes: List[Tensor] = targets["target_boxes"]
        target_classes: List[Tensor] = targets["target_classes"]
        target_seg: Tensor = targets.get("target_seg", None)

        pred_detection, anchors, pred_seg, features = self(images)

        # with torch.no_grad():
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

        # self.save_matched_anchors(images=images, target_boxes=target_boxes,
        #                             anchors=anchors, pos_idx=pos_idx,
        #                             neg_idx=neg_idx, seg=seg_targets)
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

    # TODO: refactor this with new postprocessor object
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
        boxes, probs, labels = self.postprocess_detections(
            pred_detection=pred_detection,
            anchors=anchors,
            image_shapes=image_shapes,
        )
        prediction = {"pred_boxes": boxes, "pred_scores": probs, "pred_labels": labels}

        if self.segmenter is not None:
            prediction["pred_seg"] = self.segmenter.postprocess_for_inference(pred_seg)[
                "pred_seg"
            ]
        return prediction

    def postprocess_detections(
        self,
        pred_detection: Dict[str, Tensor],
        anchors: List[Tensor],
        image_shapes: List[Tuple[int]],
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        Postprocess bounding box deltas and logits to generate final boxes and
        scores
        Adapted from torchvision https://github.com/pytorch/vision

        Args:
            pred_detection: detection predictions for loss computation

                ``"box_logits"`` Tensor
                    classification logits for each anchor [N]

                ``"box_deltas"`` Tensor
                    offsets for each anchor (x1, y1, x2, y2, (z1, z2))[N, dim * 2]

            anchors: proposals for each image
            image_shapes: shape of each image

        Returns:
            List[Tensor]: final boxes [R, dim * 2]
            List[Tensor]: final scores (for final class) [R]
            List[Tensor]: final class label [R]
        """
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in anchors]
        pred_detection = self.head.postprocess_for_inference(pred_detection, anchors)
        pred_boxes, pred_probs = (
            pred_detection["pred_boxes"],
            pred_detection["pred_probs"],
        )

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_probs = pred_probs.split(boxes_per_image, 0)

        all_boxes, all_probs, all_labels = [], [], []
        # iterate over images
        for boxes, probs, image_shape in zip(pred_boxes, pred_probs, image_shapes):
            if self.head.class_agnostic:
                _boxes, _probs, _labels = post_image_single_class_regression(
                    boxes=boxes,
                    probs=probs,
                    num_foreground_classes=self.num_foreground_classes,
                    image_shape=image_shape,
                    nms_thresh=self.nms_thresh,
                    topk_candidates=self.topk_candidates,
                    score_thresh=self.score_thresh,
                    remove_small_boxes=self.remove_small_boxes,
                    detections_per_img=self.detections_per_img,
                )
            else:
                raise NotImplementedError

            all_boxes.append(_boxes)
            all_probs.append(_probs)
            all_labels.append(_labels)
        return all_boxes, all_probs, all_labels

    # @torch.no_grad()
    # def save_matched_anchors(self, **kwargs):
    #     logger = get_logger("mllogger")
    #     logger.save_pickle("anchor_matching",
    #                        to_device(kwargs, device="cpu", detach=True))
