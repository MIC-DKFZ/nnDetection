# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from loguru import logger

from nndet.core.boxes.matcher import Matcher
from nndet.core.boxes.sampler import AbstractSampler
from nndet.core.post.box import BoxPostprocessing
from nndet.core.post.mask import MaskPostprocessing
from nndet.core.rois.module.base import BaseRoIModule
from nndet.core.rois.pooler import RoIPooler
from nndet.nn.heads.comb.roi import RoIHead
from nndet.nn.heads.masker.roi import Masker
from nndet.utils.typing import ND_TUPLE_INT


class RoIModule(BaseRoIModule):
    def __init__(
        self,
        box_head: RoIHead,
        box_pooler: RoIPooler,
        box_post: BoxPostprocessing,
        matcher: Matcher,
        sampler: AbstractSampler,
        num_classes: int,
        decoder_levels: Sequence[int],
        gt_to_proposals: bool = True,
        mask_head: Optional[Union[Masker, List[Masker], Tuple[Masker]]] = None,
        mask_pooler: Optional[RoIPooler] = None,
        mask_post: Optional[MaskPostprocessing] = None,
        inference_prob_rpn: bool = False,
    ) -> None:
        """
        RoI Module to perform RoI based detection
        "Faster R-CNN: Towards Real-Time Object Detection with Region
        Proposal Networks"
        https://arxiv.org/abs/1506.01497
        "Mask R-CNN"
        https://arxiv.org/abs/1703.06870
        (experimental) "Probabilistic two-stage detection"
        https://arxiv.org/abs/2103.07461

        Args:
            box_head: module to perform box regression and classification of
                RoIs
            box_pooler: module to perform pooling of features to be passed
                to head
            box_post: module to perform postprocessing of the box predictions
            matcher: module to assign labels to the proposal boxes
            sampler: module to sample a subset of the proposals to compute
                the loss during training
            num_classes: number of foreground classes
            decoder_levels: specify which levels should be used for the
                pooling operations
            gt_to_proposals: Add ground truth objects to the proposals during
                the training setp for improved stability at the beginning of
                the trainign. Defaults to True.
            mask_head: module to perform mask predictions of RoIs
            mask_pooler: module to perform pooling of features to be passed
                to head. Defaults to None.
            mask_post:  module to perform postprocessing of the mask
                predictions. Defaults to None.
            inference_prob_rpn: (experiental) predictions from RoI module
                are conditioned on the predictions of the RPN, which
                is realized by multiplying the predicted proposal probabilities
                with the predicted RoI probabilities.
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
            mask_head=mask_head,
            mask_pooler=mask_pooler,
            mask_post=mask_post,
            inference_prob_rpn=inference_prob_rpn,
        )
        if len(self.box_head) > 1:
            raise ValueError("Found more than one box head, use Cascade RoI head instead")
        if len(self.matcher) > 1:
            raise ValueError("Found more than one matcher, use Cascade RoI head instead")
        if self.mask_head is not None and len(self.mask_head) > 1:
            raise ValueError("Found more than one mask head, use Cascade RoI head instead")

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
        _features = [features[i] for i in self.decoder_levels]
        image_size = tuple(images.shape[2:])

        proposals = self.detach_proposals(proposals)
        (
            proposal_boxes,
            matched_gt_labels,
            matched_gt_boxes,
            matched_gt_idx,
        ) = self.assign_and_sample(proposals=proposals, targets=targets)

        if sum(pb.numel() for pb in proposal_boxes) == 0:
            logger.info(
                "No proposals found return zero loss for RoI head "
                f"with initial proposals {proposals} and targets {targets}"
            )
            return {}

        # box loss
        losses, _ = self._train_step_boxes(
            features=_features,
            matched_gt_boxes=matched_gt_boxes,
            matched_gt_labels=matched_gt_labels,
            proposal_boxes=proposal_boxes,
            image_size=image_size,
            stage=0,
            predict=False,
        )

        # mask loss
        if self.mask_mode:
            mask_losses, _ = self._train_step_masks(
                features=_features,
                matched_gt_labels=matched_gt_labels,
                matched_gt_idx=matched_gt_idx,
                proposal_boxes=proposal_boxes,
                gt_binary_masks=targets["target_binary_masks"],
                image_size=image_size,
                stage=0,
                predict=False,
            )
            losses.update(mask_losses)
        return {f"roi_s0_{k}": i for k, i in losses.items()}

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
        _features = [features[i] for i in self.decoder_levels]
        image_size = tuple(images.shape[2:])

        prediction = self._inference_step_boxes(
            features=_features,
            image_size=image_size,
            proposal_boxes=proposals["pred_boxes"],
            proposal_scores=proposals["pred_scores"],
        )

        if self.mask_mode:
            mask_preds = self._inference_step_masks(
                features=_features,
                image_size=image_size,
                pred_boxes=prediction["pred_boxes"],
                pred_probs=prediction["pred_scores"],
                pred_labels=prediction["pred_labels"],
            )
            prediction.update(mask_preds)
        return prediction
