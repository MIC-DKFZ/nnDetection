# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from loguru import logger
from torch import Tensor

import nndet.core.ops_torch as ops_torch
from nndet.core.boxes import Matcher
from nndet.core.boxes.assign import assign_targets_to_anchors
from nndet.core.boxes.sampler import AbstractSampler
from nndet.core.post.box import BoxPostprocessing
from nndet.core.post.mask import MaskPostprocessing
from nndet.core.rois.pooler import RoIPooler
from nndet.nn.heads.comb.roi import RoIHead
from nndet.nn.heads.masker.roi import Masker
from nndet.utils.tensor import cat, detach_all
from nndet.utils.typing import ND_TUPLE_INT


# TODO: cleanup
# FIXME: no proposals case -> matcher
class BaseRoIModule(torch.nn.Module):
    def __init__(
        self,
        box_head: Union[RoIHead, List[RoIHead], Tuple[RoIHead]],
        box_pooler: RoIPooler,
        box_post: BoxPostprocessing,
        matcher: Union[Matcher, List[Matcher], Tuple[Matcher]],
        sampler: AbstractSampler,  # NegativeSampler default => random balanced sampling
        num_classes: int,
        decoder_levels: Sequence[int],
        gt_to_proposals: bool = True,
        # mask
        mask_head: Optional[Union[Masker, List[Masker], Tuple[Masker]]] = None,
        mask_pooler: Optional[RoIPooler] = None,
        mask_post: Optional[MaskPostprocessing] = None,
        # post-processing
        inference_prob_rpn: bool = False,
    ) -> None:
        """
        Base class for RoI Modules

        Args:
            box_head: module to perform box regression and classification of
                RoIs; if only a single head is provided, it will be replicated
                for all stages
            box_pooler: module to perform pooling of features to be passed
                to head
            box_post: module to perform postprocessing of the box predictions
            matcher: module to assign labels to the proposal boxes. if only a
                single head is provided, it will be replicated for all stages
            sampler: module to sample a subset of the proposals to compute
                the loss during training
            num_classes: number of foreground classes
            decoder_levels: specify which levels should be used for the
                pooling operations
            gt_to_proposals: Add ground truth objects to the proposals during
                the training setp for improved stability at the beginning of
                the trainign. Defaults to True.
            mask_head: module to perform mask predictions of RoIs;
                if only a single head is provided, it will be replicated
                for all stages. Defaults to None.
            mask_pooler: module to perform pooling of features to be passed
                to head. Defaults to None.
            mask_post:  module to perform postprocessing of the mask
                predictions. Defaults to None.
            inference_prob_rpn: (experiental) predictions from RoI module
                are conditioned on the predictions of the RPN, which
                is realized by multiplying the predicted proposal probabilities
                with the predicted RoI probabilities.
        """
        super().__init__()
        # Box Setup
        if not isinstance(box_head, (list, tuple)):
            box_head = [box_head]
        if not isinstance(matcher, (list, tuple)):
            matcher = [matcher]
        self.num_stages = len(box_head)

        if len(box_head) != len(matcher):
            raise ValueError(
                f"Each stage needs to have a matcher and box head. "
                f"Received {len(box_head)} box_heads and {len(matcher)} matchers"
            )

        self.box_head = torch.nn.ModuleList(list(box_head))
        self.box_pooler = box_pooler
        self.box_post = box_post

        self.matcher = matcher
        self.sampler = sampler

        self.num_foreground_classes = num_classes
        self.decoder_levels = decoder_levels
        self.gt_to_proposals = gt_to_proposals

        # Mask Setup
        if mask_head is not None and mask_pooler is None:
            raise ValueError("Mask mode requires head and pooler to be set! " "Mask Pooler was not porovided.")
        if mask_pooler is not None and mask_head is None:
            raise ValueError("Mask mode requires head and pooler to be set! " "Mask Head was not porovided.")
        self.mask_mode = mask_head is not None and mask_pooler is not None
        if self.mask_mode:
            logger.info("Running mask branch for training")
            if not isinstance(mask_head, (list, tuple)):
                mask_head = [mask_head]
            if len(mask_head) != self.num_stages:
                raise ValueError(
                    f"Each stage needs to have a matcher and box head. "
                    f"Received {len(mask_head)} mask heads but has {self.num_stages} stages."
                )
            if mask_post is None:
                raise ValueError("Need to provide mask postprocessing in mask mode.")

            self.mask_head = torch.nn.ModuleList(list(mask_head))
        else:
            self.mask_head = None
        self.mask_pooler = mask_pooler
        self.mask_post = mask_post

        self.inference_prob_rpn = inference_prob_rpn
        if self.inference_prob_rpn:
            logger.warning("'inference_prob_rpn' is an experimental setting it might not work correctly.")

        # logging
        logger.info(f"RoI Module: gt_to_proposals {self.gt_to_proposals}")
        logger.info(f"RoI Module: inference_prob_rpn {self.inference_prob_rpn}")

    @abstractmethod
    def train_step(
        self,
        images: torch.Tensor,
        features: List[torch.Tensor],
        proposals: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        targets: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a training step of the RoI Module

        Args:
            images: batch of input images
            features: multi-scale features from neck
            proposals: proposals, usually from Region Proposal Network

                ``"pred_boxes"`` List[Tensor]
                    proposed boxes [N, dims * 2]
                    (x_min, y_min, x_max, y_max, z_min, z_max)

                ``"pred_scores"`` List[Tensor]
                    associated scores for each proposal [N]

                ``"pred_labels"`` List[Tensor]
                    associated label for each proposal [N]

            targets: ground truth
                ``"target_boxes"`` List[Tensor]
                    ground truth boxes [R, dims * 2]
                    (x_min, y_min, x_max, y_max, z_min, z_max)

                ``"target_roi_classes"`` List[Tensor]
                    associated class for each ground truth object [R]

                ``"target_binary_masks"`` List[Tensor]
                    Only required when additional mask head is provided.
                    associated binary mask for each ground truth object
                    [R, image_size]. The i-th entry along the first dimension
                    corresponds to the i-th object / box / class.

        Returns:
            Dict[str, torch.Tensor]: computed losses
        """
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def inference_step(
        self,
        images: torch.Tensor,
        features: List[torch.Tensor],
        proposals: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform an inference step of the RoI Module

        Args:
            images: batch of input images
            features: multi-scale features from neck
            proposals: proposals, usually from Region Proposal Network

                ``"pred_boxes"`` List[Tensor]
                    proposed boxes [N, dims * 2]
                    (x_min, y_min, x_max, y_max, z_min, z_max)

                ``"pred_scores"`` List[Tensor]
                    associated scores for each proposal [N]

                ``"pred_labels"`` List[Tensor]
                    associated label for each proposal [N]

            kwargs: ignored

        Returns:
            Dict[str, Any]: predictions

                ``"pred_boxes"`` List[Tensor]
                    predicted boxes [N, dims * 2]
                    (x_min, y_min, x_max, y_max, z_min, z_max)

                ``"pred_scores"`` List[Tensor]
                    associated scores for each predicted box [N]

                ``"pred_labels"`` List[Tensor]
                    associated labels for each predicted box [N]

                ``"pred_masks"`` List[Tensor]
                    predicted probability masks from mask head [N, RoI_dims]
                    The output size of the masks are determined by the Masker
                    RoI Module. To compute the evaluated additional post-
                    processing will be required.

                ``"pred_mask_scores"`` List[Tensor]
                    associated scores for each predicted masks [N]

                ``"pred_mask_labels"`` List[Tensor]
                    associated labels for each predicted masks [N]

                ``"pred_image_spatial_size"`` ND_TUPLE_INT
                    image size which was used for prediction. Needed to restore
                    correct size of image when pasting binary masks.

        """
        raise NotImplementedError

    def _train_step_boxes(
        self,
        features: List[Tensor],
        image_size: ND_TUPLE_INT,
        matched_gt_boxes: List[Tensor],
        matched_gt_labels: List[Tensor],
        proposal_boxes: List[Tensor],
        stage: int = 0,
        predict: bool = False,
    ) -> Tuple[Dict[str, Tensor], Optional[Dict[str, List[Tensor]]]]:
        """
        Compute losses for Box Head

        Args:
            features: multi-scale features which should be used for pooling
            matched_gt_boxes: ground truth boxes matched to proposals
                List[[N, dim * 2]] where N is the number of (sampled) proposals
                in (x_min, y_min, x_max, y_max, z_min, z_max) format
            matched_gt_labels: ground truth labels matched to proposals
                List[N] where N is the number of (sampled) proposals
            proposal_boxes: proposal boxes
                List[[N, dim * 2]] where N is the number of proposals
                in (x_min, y_min, x_max, y_max, z_min, z_max) format
            image_size: image size
            stage: Current stage to predict. Defaults to 0.
            predict: Perform postprocessing of subsampled predictions.
                Note, if `gt_to_proposals` is set to True, this will
                contain the ground truth boxes as proposals and thus
                also as predictions. Defaults to False.

        Returns:
            Dict[str, Tensor]: losses
            Optional[Dict[str, List[Tensor]]]: None is `predict=False`
                otherwise is contains the predictions

                ``"pred_boxes"`` List[Tensor]
                    predicted boxes [N, dims * 2]
                    (x_min, y_min, x_max, y_max, z_min, z_max)

                ``"pred_scores"`` List[Tensor]
                    associated scores for each predicted box [N]

                ``"pred_labels"`` List[Tensor]
                    associated labels for each predicted box [N]
        """
        batch_size = len(proposal_boxes)
        matched_gt_labels = cat(matched_gt_labels)
        matched_gt_boxes = cat(matched_gt_boxes)
        _proposal_boxes, batch_idx = ops_torch.cat_and_index(proposal_boxes)

        box_roi_features = self.box_pooler(
            features=features,
            proposal_boxes=_proposal_boxes,
            batch_idx=batch_idx,
            image_size=image_size,
        )  # [N, C, spatial]; N=num proposals passed, C=number of feature channels

        pred_detection = self.box_head[stage](box_roi_features)
        losses, _, _ = self.box_head[stage].compute_loss(
            prediction=pred_detection,
            matched_gt_labels=matched_gt_labels,
            matched_gt_boxes=matched_gt_boxes,
            proposal_boxes=_proposal_boxes,
        )

        if predict:
            image_shapes = [image_size] * batch_size
            boxes, probs, labels = self.postprocess_detections(
                pred_detection=pred_detection,
                proposal_boxes=proposal_boxes,
                image_shapes=image_shapes,
                stage=stage,
            )
            prediction = {
                "pred_boxes": boxes,
                "pred_scores": probs,
                "pred_labels": labels,
            }
        else:
            prediction = None
        return losses, prediction

    def _train_step_masks(
        self,
        features: List[Tensor],
        matched_gt_labels: List[Tensor],
        matched_gt_idx: List[Tensor],
        proposal_boxes: List[Tensor],
        gt_binary_masks: List[Tensor],
        image_size: ND_TUPLE_INT,
        stage: int = 0,
        predict: bool = False,
    ) -> Tuple[Dict[str, Tensor], None]:
        """
        Compute losses for Mask Head

        Args:
            features: multi-scale features which should be used for pooling
            matched_gt_labels: ground truth labels matched to proposals
                List[N] where N is the number of (sampled) proposals
            matched_gt_idx: index of ground truth matched to proposals
                List[N] where N is the number of (sampled) proposals
            proposal_boxes: proposal boxes
                List[[N, dim * 2]] where N is the number of proposals
                in (x_min, y_min, x_max, y_max, z_min, z_max) format
            gt_binary_masks: ground truth binary masks
                List[[X, dims]] where X is the number of ground truth objects
                    in the image and dims are spatial dimensions
            image_size: image size
            stage: Current stage to predict. Defaults to 0.
            predict: Perform postprocessing of subsampled predictions.
                Note, if `gt_to_proposals` is set to True, this will
                contain the ground truth boxes as proposals and thus
                also as predictions. Defaults to False.

        Returns:
            Dict[str, Tensor]: losses
            Optional[Dict[str, List[Tensor]]]: None. Kept for consistency
                of steps
        """
        # compute mask loss on positive proposals
        pos_matched_gt_idx = []
        pos_proposal_boxes = []
        pos_label = []
        for gt_l, gt_idx, prop_b in zip(matched_gt_labels, matched_gt_idx, proposal_boxes):
            pos_idx = torch.where(gt_l > 0)[0]
            pos_matched_gt_idx.append(gt_idx[pos_idx])
            pos_proposal_boxes.append(prop_b[pos_idx])
            pos_label.append(gt_l[pos_idx])

        target_masks_prepared = self.mask_pooler.pool_masks(
            binary_masks=gt_binary_masks,
            proposal_boxes=pos_proposal_boxes,
            matched_gt_idx=pos_matched_gt_idx,
        )  # List[[R, output_size]]

        pos_proposal_boxes, batch_idx = ops_torch.cat_and_index(pos_proposal_boxes)

        mask_roi_features = self.mask_pooler(
            features=features,
            proposal_boxes=pos_proposal_boxes,
            batch_idx=batch_idx,
            image_size=image_size,
        )  # [N, C, spatial]; N=num proposals passed, C=number of feature channels

        pred_masks, _ = self.mask_head[stage](mask_roi_features)
        target_masks_prepared_batched = torch.cat(target_masks_prepared, dim=0)
        assert pred_masks.shape[0] == target_masks_prepared_batched.shape[0]
        # TODO: check for consistency
        batch_pos_label = torch.cat(pos_label) - 1
        assert batch_pos_label.shape[0] == pred_masks.shape[0]
        losses = self.mask_head[stage].compute_loss(
            pred_logits=pred_masks,
            target_masks=target_masks_prepared_batched,
            target_labels=batch_pos_label,
        )

        return losses, None

    def detach_proposals(
        self,
        proposals: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Don't propagate gradients through the proposals.

        Args:
            proposals: dict with proposals

        Returns:
            Dict[str, Union[torch.Tensor, List[torch.Tensor]]]: detached
                proposals
        """
        return detach_all(proposals)

    @torch.no_grad()
    def assign_and_sample(
        self,
        proposals: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        targets: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        """
        Assign boxes and labels to proposals and sample a supset of them
        to compute the loss. If `gt_to_proposals=True` ground truth
        objects are added to the proposals to (potentially) improve the
        training stability in the beginning of the training.

        Args:
            proposals: proposals, usually from Region Proposal Network

                ``"pred_boxes"`` List[Tensor]
                    proposed boxes [N, dims * 2]
                    (x_min, y_min, x_max, y_max, z_min, z_max)

                ``"pred_scores"`` List[Tensor]
                    associated scores for each proposal [N]

                ``"pred_labels"`` List[Tensor]
                    associated label for each proposal [N]

            targets: ground truth
                ``"target_boxes"`` List[Tensor]
                    ground truth boxes [R, dims * 2]
                    (x_min, y_min, x_max, y_max, z_min, z_max)

                ``"target_roi_classes"`` List[Tensor]
                    associated class for each ground truth object [R]

        Returns:
            List[Tensor]: proposal boxes
                List[[N, dim * 2]] where N is the number of (sampled)
                proposals in (x_min, y_min, x_max, y_max, z_min, z_max)
                format
            List[Tensor]: ground truth labels matched to proposals
                List[N] where N is the number of (sampled) proposals
            List[Tensor]: ground truth boxes matched to proposals
                List[[N, dim * 2]] where N is the number of (sampled)
                proposals in (x_min, y_min, x_max, y_max, z_min, z_max)
                format
            List[Tensor]: index of ground truth matched to proposals
                List[N] where N is the number of (sampled) proposals
        """
        # optionally add proposals to prediction pool
        if self.gt_to_proposals:
            proposals = self.add_gt_to_proposals(proposals, targets)

        # match proposals to ground truth
        matched_gt_labels, matched_gt_boxes, matched_gt_idx = assign_targets_to_anchors(
            proposal_matcher=self.matcher[0],
            anchors=proposals["pred_boxes"],
            target_boxes=targets["target_boxes"],
            target_classes=targets["target_roi_classes"],
        )  # List([N]), List([N, dims * 2]), List([N])

        # sample proposals and gt
        pos_mask, neg_mask = self.sampler(
            target_labels=matched_gt_labels,
            fg_probs=proposals["pred_scores"],
        )  # List([N]), List([N])

        proposal_boxes_sampled = []
        matched_gt_labels_sampled = []
        matched_gt_boxes_sampled = []
        matched_gt_idx_sampled = []
        for img_idx, (pm, nm) in enumerate(zip(pos_mask, neg_mask)):
            sampled_pos_inds = torch.where(pm)[0]
            sampled_neg_inds = torch.where(nm)[0]
            inds = cat([sampled_pos_inds, sampled_neg_inds], dim=0)

            proposal_boxes_sampled.append(proposals["pred_boxes"][img_idx][inds])
            matched_gt_labels_sampled.append(matched_gt_labels[img_idx][inds])
            matched_gt_boxes_sampled.append(matched_gt_boxes[img_idx][inds])
            matched_gt_idx_sampled.append(matched_gt_idx[img_idx][inds])
        return (
            proposal_boxes_sampled,
            matched_gt_labels_sampled,
            matched_gt_boxes_sampled,
            matched_gt_idx_sampled,
        )

    def add_gt_to_proposals(
        self,
        proposals: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        targets: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Add ground truth boxes to proposals.
        This helps training in the early stages when proposals are bad.

        Args:
            proposals: box proposals for each image
                List[[N, dim * 2]], N=number of proposals per image
            gt: ground truth boxes for each image
                List[[N, dim * 2]], N=number of ground truth per image

        Returns:
            List[torch.Tensor]: proposals with gt boxes
                List[[N + M, dim * 2]], N + M = new number of proposals
        """
        for i in range(len(targets["target_boxes"])):
            if targets["target_boxes"][i].numel() > 0:
                proposals["pred_boxes"][i] = cat([proposals["pred_boxes"][i], targets["target_boxes"][i]], dim=0)

                tc = targets["target_roi_classes"][i]
                proposals["pred_labels"][i] = cat([proposals["pred_labels"][i], tc], dim=0)

                add_scores = torch.ones(
                    tc.shape[0],
                    device=proposals["pred_scores"][i].device,
                    dtype=proposals["pred_scores"][i].dtype,
                )
                proposals["pred_scores"][i] = cat([proposals["pred_scores"][i], add_scores], dim=0)
        return proposals

    def _inference_step_boxes(
        self,
        features: List[torch.Tensor],
        image_size: ND_TUPLE_INT,
        proposal_boxes: List[torch.Tensor],
        proposal_scores: Optional[List[torch.Tensor]] = None,
        stage: int = 0,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Perform Inference of Box Head

        Args:
            features: multi-scale features which should be used for pooling
            image_size: image size
            proposal_boxes: proposal boxes
                List[[N, dim * 2]] where N is the number of proposals
                in (x_min, y_min, x_max, y_max, z_min, z_max) format
            proposal_scores: proposal scores (/probabilities)
                List[N] where N is the number of proposals
            stage: Current stage to predict. Defaults to 0.

        Returns:
            Dict[str, List[torch.Tensor]]: None is `predict=False`
                otherwise is contains the predictions

                ``"pred_boxes"`` List[Tensor]
                    predicted boxes [N, dims * 2]
                    (x_min, y_min, x_max, y_max, z_min, z_max)

                ``"pred_scores"`` List[Tensor]
                    associated scores for each predicted box [N]

                ``"pred_labels"`` List[Tensor]
                    associated labels for each predicted box [N]
        """
        _proposal_boxes, batch_idx = ops_torch.cat_and_index(proposal_boxes)
        batch_size = len(proposal_boxes)

        if _proposal_boxes.numel() == 0:
            dtype = proposal_boxes[0].dtype
            device = proposal_boxes[0].device
            boxes = [torch.zeros_like(proposal_boxes[b]) for b in range(batch_size)]
            probs = [torch.tensor([], dtype=dtype, device=device) for b in range(batch_size)]
            labels = [torch.tensor([], dtype=torch.int64, device=device) for b in range(batch_size)]
        else:
            roi_features = self.box_pooler(
                features=features,
                proposal_boxes=_proposal_boxes,
                batch_idx=batch_idx,
                image_size=image_size,
            )  # [P, C, spatial]

            pred_detection = self.box_head[stage](roi_features)
            boxes, probs, labels = self.postprocess_detections(
                pred_detection=pred_detection,
                proposal_boxes=proposal_boxes,
                image_shapes=[image_size] * batch_size,
                stage=stage,
                proposal_scores=proposal_scores,
                apply_inference_prob_rpn=self.inference_prob_rpn,
            )
        prediction = {
            "pred_boxes": boxes,
            "pred_scores": probs,
            "pred_labels": labels,
        }
        return prediction

    def _inference_step_masks(
        self,
        features: List[torch.Tensor],
        image_size: ND_TUPLE_INT,
        pred_boxes: List[torch.Tensor],
        pred_probs: List[torch.Tensor],
        pred_labels: List[torch.Tensor],
        stage: int = 0,
    ) -> Dict[str, List[Tensor]]:
        """
        Perform Inference of Mask Head

        Args:
            features: multi-scale features which should be used for pooling
            image_size: image size
            pred_boxes: predicted boxes
                List[[N, dim * 2]] where N is the number of prediction
                in (x_min, y_min, x_max, y_max, z_min, z_max) format
            pred_probs: predicted scores (/probabilities)
                List[N] where N is the number of predictions
            pred_labels: predicted labels
                List[N] where N is the number of predictions
            stage: Current stage to predict. Defaults to 0.

        Returns:
            Dict[str, List[torch.Tensor]]: None is `predict=False`
                otherwise is contains the predictions

                ``"pred_masks"`` List[Tensor]
                    predicted probability masks from mask head [N, RoI_dims]
                    The output size of the masks are determined by the Masker
                    RoI Module. To compute the evaluated additional post-
                    processing will be required.

                ``"pred_mask_scores"`` List[Tensor]
                    associated scores for each predicted masks [N]

                ``"pred_mask_labels"`` List[Tensor]
                    associated labels for each predicted masks [N]

                ``"pred_image_spatial_size"`` ND_TUPLE_INT
                    image size which was used for prediction. Needed to restore
                    correct size of image when pasting binary masks.
        """
        _boxes, batch_idx = ops_torch.cat_and_index(pred_boxes)
        batch_size = len(pred_boxes)

        if _boxes.numel() == 0:
            dtype = pred_boxes[0].dtype
            device = pred_boxes[0].device
            masks = [torch.tensor([], dtype=dtype, device=device) for b in range(batch_size)]
            probs = [torch.tensor([], dtype=dtype, device=device) for b in range(batch_size)]
            labels = [torch.tensor([], dtype=torch.int64, device=device) for b in range(batch_size)]
        else:
            roi_features = self.mask_pooler(
                features=features,
                proposal_boxes=_boxes,
                batch_idx=batch_idx,
                image_size=image_size,
            )  # [P, C, spatial]

            _masks, _ = self.mask_head[stage](roi_features)  # [P, C, mask_dim]

            masks, probs, labels = self.postprocess_masks(
                masks=_masks,
                pred_probs=pred_probs,
                pred_labels=pred_labels,
                image_shapes=[image_size] * batch_size,
                stage=stage,
            )
        prediction = {
            "pred_masks": masks,
            "pred_mask_scores": probs,
            "pred_mask_labels": labels,
            "pred_image_spatial_size": image_size,
        }
        return prediction

    @torch.no_grad()
    def postprocess_detections(
        self,
        pred_detection: Dict[str, torch.Tensor],
        proposal_boxes: List[torch.Tensor],
        image_shapes: List[Tuple[int]],
        stage: int,
        proposal_scores: Optional[List[torch.Tensor]] = None,
        apply_inference_prob_rpn: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Perform postprocessing of box predictions

        Args:
            pred_detection: predicted deltas and logits by head

                ``'box_deltas'`` (Tensor)
                    bounding box offsets
                    [num_proposals, (num_classes), dim * 2];
                    num classes is only present if anchors were regressed
                    for each class individually

                ``'box_logits'`` (Tensor)
                    classification logits [num_proposals, num_classes]

            proposal_boxes: proposal boxes
                List[[N, dim * 2]] where N is the number of proposals
                in (x_min, y_min, x_max, y_max, z_min, z_max) format
            image_shapes: shape of each image
            stage: Current stage to predict. Defaults to 0.
            proposal_scores: proposal scores. Defaults to None.
                List[N] where N is the number of proposals
            apply_inference_prob_rpn: Multiply roi scores predictions
                with scores generated by RPN. Defaults to False.

        Returns:
            List[torch.Tensor]: predicted boxes [N, dims * 2] in
                (x_min, y_min, x_max, y_max, z_min, z_max) format
            List[torch.Tensor]: associated scores for each predicted box [N];
                N is the number of proposals/RoIs
            List[torch.Tensor]: associated labels for each predicted box [N];
                N is the number of proposals/RoIs
        """
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposal_boxes]

        pred_detection = self.box_head[stage].postprocess_for_inference(pred_detection, proposal_boxes)
        pred_boxes, pred_probs = (
            pred_detection["pred_boxes"],
            pred_detection["pred_probs"],
        )

        if apply_inference_prob_rpn:
            assert proposal_scores is not None
            pred_probs = pred_probs * cat(proposal_scores).unsqueeze_(-1)

        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_probs = pred_probs.split(boxes_per_image, 0)

        return self.box_post.process_batch(
            reps=pred_boxes,
            probs=pred_probs,
            image_shapes=image_shapes,
        )

    @torch.no_grad()
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        pred_probs: List[torch.Tensor],
        pred_labels: List[torch.Tensor],
        image_shapes: List[Tuple[int]],
        stage: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Perform postprocessing of box predictions

        Args:
            masks: predicted masks [N, num_classes, dims], where N is the
                number of RoIs, num_classes is the number of foreground classes
                and dims are spatial dimensions
            pred_probs: predicted probabilities for each mask
                List[N] where N is the number of predictions/RoIs
            pred_labels: predicted label for each mask
                List[N] where N is the number of predictions/RoIs
            image_shapes: shape of each image
            stage: Current stage to predict. Defaults to 0.

        Returns:
            List[torch.Tensor]: predicted boxes [N, dims * 2] in
                (x_min, y_min, x_max, y_max, z_min, z_max) format
            List[torch.Tensor]: associated scores for each predicted box [N];
                N is the number of proposals/RoIs
            List[torch.Tensor]: associated labels for each predicted box [N];
                N is the number of proposals/RoIs
        """
        masks_per_image = [len(pl) for pl in pred_labels]
        assert [len(pp) == len(pl) for pp, pl in zip(pred_probs, pred_labels)]
        assert sum(masks_per_image) == masks.shape[0]
        batched_mask_labels = torch.cat(pred_labels)

        pred_masks = self.mask_head[stage].logits_to_probs(masks, batched_mask_labels)
        pred_masks = pred_masks.split(masks_per_image, 0)

        return self.mask_post.process_batch(
            reps=pred_masks,
            probs=pred_probs,
            labels=pred_labels,
        )
