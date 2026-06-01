# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from loguru import logger

from nndet.core.boxes import Matcher
from nndet.core.boxes.assign import assign_targets_to_anchors
from nndet.core.boxes.sampler import AbstractSampler
from nndet.core.post.box import BoxPostprocessing
from nndet.core.post.mask import MaskPostprocessing
from nndet.core.rois.module.base import BaseRoIModule
from nndet.core.rois.pooler import RoIPooler
from nndet.nn.heads.comb.base import RoIHead
from nndet.nn.heads.masker.roi import Masker
from nndet.utils.typing import ND_TUPLE_INT


class CascadeRoIModule(BaseRoIModule):
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
        mask_interleaved_execution: bool = False,
        # post-processing
        # no postprocessing option here
        # cascade settings
        loss_weight_stage: Optional[Sequence[float]] = None,
    ) -> None:
        """
        RoI Module to perform multiple sequential/cascaded RoI based detections
        "Cascade R-CNN: Delving into High Quality Object Detection"
        https://arxiv.org/abs/1712.00726
        "Hybrid Task Cascade for Instance Segmentation"
        https://arxiv.org/abs/1901.07518

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
            mask_interleaved_execution: Interleaved mask execution as proposed
                in HTC. Defaults to False.
            loss_weight_stage: Provide loss weights for each stage to scale
                the losses. Defaults to None.

        Raises:
            ValueError: Need to provide loss weight for each stage
        """
        super().__init__(
            box_head=box_head,
            box_pooler=box_pooler,
            box_post=box_post,
            matcher=matcher,
            sampler=sampler,
            num_classes=num_classes,
            decoder_levels=decoder_levels,
            gt_to_proposals=gt_to_proposals,
            # mask
            mask_head=mask_head,
            mask_pooler=mask_pooler,
            mask_post=mask_post,
        )
        if loss_weight_stage is None:
            self.loss_weight_stage = [1.0] * self.num_stages
        else:
            if len(loss_weight_stage) != self.num_stages:
                raise ValueError(
                    "If loss weight is provided, each stage needs one! "
                    f"Found {loss_weight_stage} but {self.num_stages} stages."
                )
            self.loss_weight_stage = list(map(float, loss_weight_stage))
        logger.info(f"Running Cascade RoI Module with {self.num_stages} stages and " f"train mask {self.mask_mode}")
        self.mask_interleaved_execution = mask_interleaved_execution

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
        fpn_features = [features[i] for i in self.decoder_levels]
        image_size = tuple(images.shape[2:])

        proposals = self.detach_proposals(proposals)
        (
            proposal_boxes,
            matched_gt_labels,
            matched_gt_boxes,
            matched_gt_idx,
        ) = self.assign_and_sample(proposals=proposals, targets=targets)

        losses = {}
        for stage_idx in range(self.num_stages):
            if stage_idx > 0:
                proposals = self.detach_proposals(new_proposals)  # noqa: F821
                proposal_boxes = proposals["pred_boxes"]
                # match proposals to ground truth
                (matched_gt_labels, matched_gt_boxes, matched_gt_idx,) = assign_targets_to_anchors(
                    proposal_matcher=self.matcher[stage_idx],
                    anchors=proposal_boxes,
                    target_boxes=targets["target_boxes"],
                    target_classes=targets["target_roi_classes"],
                )  # List([N]), List([N, dims * 2]), List([N])

            # box loss
            box_losses, new_proposals = self._train_step_boxes(
                features=fpn_features,
                matched_gt_boxes=matched_gt_boxes,
                matched_gt_labels=matched_gt_labels,
                proposal_boxes=proposal_boxes,
                image_size=image_size,
                stage=stage_idx,
                predict=True,
            )
            for k, i in box_losses.items():
                losses[f"roi_s{stage_idx}_{k}"] = i * self.loss_weight_stage[stage_idx]

            # mask loss
            if self.mask_mode:
                if self.mask_interleaved_execution:  # use new boxes for mask
                    proposals = self.detach_proposals(new_proposals)
                    proposal_boxes = proposals["pred_boxes"]
                    (matched_gt_labels, matched_gt_boxes, matched_gt_idx,) = assign_targets_to_anchors(
                        proposal_matcher=self.matcher[stage_idx],
                        anchors=proposal_boxes,
                        target_boxes=targets["target_boxes"],
                        target_classes=targets["target_roi_classes"],
                    )  # List([N]), List([N, dims * 2]), List([N])

                mask_losses, _ = self._train_step_masks(
                    features=fpn_features,
                    matched_gt_labels=matched_gt_labels,
                    matched_gt_idx=matched_gt_idx,
                    proposal_boxes=proposal_boxes,
                    gt_binary_masks=targets["target_binary_masks"],
                    image_size=image_size,
                    stage=stage_idx,
                    predict=False,
                )
                for k, i in mask_losses.items():
                    losses[f"roi_s{stage_idx}_{k}"] = i * self.loss_weight_stage[stage_idx]
        return losses

    @torch.no_grad()
    def inference_step(
        self,
        images: torch.Tensor,
        features: List[torch.Tensor],
        proposals: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        **kwargs,
    ) -> Dict[str, Union[List[torch.Tensor], torch.Tensor, ND_TUPLE_INT]]:
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
        fpn_features = [features[i] for i in self.decoder_levels]
        image_size = tuple(images.shape[2:])

        for stage_idx in range(self.num_stages):
            if stage_idx > 0:
                proposals = prediction  # noqa: F821

            prediction = self._inference_step_boxes(
                features=fpn_features,
                image_size=image_size,
                proposal_boxes=proposals["pred_boxes"],
                stage=stage_idx,
            )
            if self.mask_mode:
                if self.mask_interleaved_execution:
                    proposal_boxes = prediction["pred_boxes"]
                    proposal_probs = prediction["pred_scores"]
                    proposal_labels = prediction["pred_labels"]
                else:
                    proposal_boxes = proposals["pred_boxes"]
                    proposal_probs = proposals["pred_scores"]
                    proposal_labels = proposals["pred_labels"]

                mask_preds = self._inference_step_masks(
                    features=fpn_features,
                    image_size=image_size,
                    pred_boxes=proposal_boxes,
                    pred_probs=proposal_probs,
                    pred_labels=proposal_labels,
                    stage=stage_idx,
                )
                prediction.update(mask_preds)
        return prediction
