from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from loguru import logger

from nndet.arch.heads.comb import RoIHeadType
from nndet.arch.heads.masker.base import MaskerType
from nndet.core.boxes import MatcherType
from nndet.core.boxes.assign import assign_targets_to_anchors
from nndet.core.boxes.sampler import SamplerType
from nndet.core.post.box import BoxPostprocessing
from nndet.core.post.mask import MaskPostprocessing
from nndet.core.rois.module.base import BaseRoIModule
from nndet.core.rois.pooler import RoIPooler

# TODO: cleanup


class CascadeRoIModule(BaseRoIModule):
    def __init__(
        self,
        box_head: Union[RoIHeadType, List[RoIHeadType], Tuple[RoIHeadType]],
        box_pooler: RoIPooler,
        box_post: BoxPostprocessing,
        matcher: Union[MatcherType, List[MatcherType], Tuple[MatcherType]],
        sampler: SamplerType,  # NegativeSampler default => random balanced sampling
        num_classes: int,
        decoder_levels: Sequence[int],
        gt_to_proposals: bool = True,
        # mask
        mask_head: Optional[
            Union[MaskerType, List[MaskerType], Tuple[MaskerType]]
        ] = None,
        mask_pooler: Optional[RoIPooler] = None,
        mask_post: Optional[MaskPostprocessing] = None,
        mask_interleaved_execution: bool = False,
        # post-processing
        roi_score_thresh: float = None,
        roi_detections_per_img: int = 100,
        roi_nms_thresh: float = 0.6,
        # cascade settings
        loss_weight_stage: Optional[Sequence[float]] = None,
    ) -> None:
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
            # post-processing
            roi_score_thresh=roi_score_thresh,
            roi_detections_per_img=roi_detections_per_img,
            roi_nms_thresh=roi_nms_thresh,
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
        logger.info(
            f"Running Cascade RoI Module with {self.num_stages} stages and "
            f"train mask {self.mask_mode}"
        )
        self.mask_interleaved_execution = mask_interleaved_execution

    def train_step(
        self,
        images: torch.Tensor,
        features: List[torch.Tensor],
        proposals: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        targets: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    ):
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
            if stage_idx != 0:
                proposals = self.detach_proposals(new_proposals)  # noqa: F821
                proposal_boxes = proposals["pred_boxes"]
                # match proposals to ground truth
                (
                    matched_gt_labels,
                    matched_gt_boxes,
                    matched_gt_idx,
                ) = assign_targets_to_anchors(
                    proposal_matcher=self.matcher[0],
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
                    (
                        matched_gt_labels,
                        matched_gt_boxes,
                        matched_gt_idx,
                    ) = assign_targets_to_anchors(
                        proposal_matcher=self.matcher[0],
                        anchors=proposal_boxes,
                        target_boxes=targets["target_boxes"],
                        target_classes=targets["target_roi_classes"],
                    )  # List([N]), List([N, dims * 2]), List([N])

                mask_losses, _ = self._train_step_masks(
                    features=fpn_features,
                    matched_gt_labels=matched_gt_labels,
                    matched_gt_idx=matched_gt_idx,
                    proposal_boxes=proposal_boxes,
                    target_binary_masks=targets["target_binary_masks"],
                    image_size=image_size,
                    stage=stage_idx,
                    predict=False,
                )
                for k, i in mask_losses.items():
                    losses[f"roi_s{stage_idx}_{k}"] = (
                        i * self.loss_weight_stage[stage_idx]
                    )
        return losses

    @torch.no_grad()
    def inference_step(
        self,
        images: torch.Tensor,
        features: List[torch.Tensor],
        proposals: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        **kwargs,
    ) -> Dict[str, Any]:
        fpn_features = [features[i] for i in self.decoder_levels]

        for stage_idx in range(self.num_stages):
            if stage_idx != 0:
                proposals = prediction  # noqa: F821

            prediction = self._inference_step_boxes(
                images=images,
                features=fpn_features,
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
                    images=images,
                    features=fpn_features,
                    pred_boxes=proposal_boxes,
                    pred_probs=proposal_probs,
                    pred_labels=proposal_labels,
                    stage=stage_idx,
                )
                prediction.update(mask_preds)
        return prediction
