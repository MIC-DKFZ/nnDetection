from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import torch
from loguru import logger
from torch import Tensor

from nndet.arch.heads.comb import RoIHeadType
from nndet.arch.heads.masker.base import MaskerType
from nndet.core.boxes import MatcherType
from nndet.core.boxes.assign import assign_targets_to_anchors
from nndet.core.boxes.ops import cat_and_index
from nndet.core.boxes.post import post_image_single_class_regression
from nndet.core.boxes.sampler import SamplerType
from nndet.core.rois.ops import create_binary_masks
from nndet.core.rois.pooler import NDSIZE, PoolerType
from nndet.utils.tensor import cat, detach_all


# TODO: cleanup
# FIXME: no proposals case -> matcher
class BaseRoIModule(torch.nn.Module):
    def __init__(
        self,
        box_head: Union[RoIHeadType, List[RoIHeadType], Tuple[RoIHeadType]],
        box_pooler: PoolerType,
        matcher: Union[MatcherType, List[MatcherType], Tuple[MatcherType]],
        sampler: SamplerType,  # NegativeSampler default => random balanced sampling
        num_classes: int,
        decoder_levels: Sequence[int],
        gt_to_proposals: bool = True,
        # mask
        mask_head: Optional[
            Union[MaskerType, List[MaskerType], Tuple[MaskerType]]
        ] = None,
        mask_pooler: Optional[PoolerType] = None,
        # post-processing
        roi_score_thresh: float = None,
        roi_detections_per_img: int = 100,
        roi_nms_thresh: float = 0.6,
    ) -> None:
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

        self.matcher = matcher
        self.sampler = sampler

        self.num_foreground_classes = num_classes
        self.decoder_levels = decoder_levels
        self.gt_to_proposals = gt_to_proposals

        # Mask Setup
        if mask_head is not None and mask_pooler is None:
            raise ValueError(
                "Mask mode requires head and pooler to be set! "
                "Mask Pooler was not porovided."
            )
        if mask_pooler is not None and mask_head is None:
            raise ValueError(
                "Mask mode requires head and pooler to be set! "
                "Mask Head was not porovided."
            )
        self.mask_mode_train = mask_head is not None and mask_pooler is not None
        if self.mask_mode_train:
            logger.info("Running mask branch for training")
            if not isinstance(mask_head, (list, tuple)):
                mask_head = [mask_head]
            if len(mask_head) != self.num_stages:
                raise ValueError(
                    f"Each stage needs to have a matcher and box head. "
                    f"Received {len(mask_head)} mask heads but has {self.num_stages} stages."
                )

            self.mask_head = torch.nn.ModuleList(list(mask_head))
            self.mask_pooler = mask_pooler

        # Inference
        self.roi_score_thresh = roi_score_thresh
        self.roi_detections_per_img = roi_detections_per_img
        self.roi_nms_thresh = roi_nms_thresh

    @abstractmethod
    def train_step(
        self,
        images: torch.Tensor,
        features: List[torch.Tensor],
        proposals: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        targets: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        predict: bool = False,
    ):
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
        raise NotImplementedError

    def _train_step_boxes(
        self,
        features: List[Tensor],
        matched_gt_boxes: List[Tensor],
        matched_gt_labels: List[Tensor],
        proposal_boxes: List[Tensor],
        image_size: NDSIZE,
        stage: int = 0,
        predict: bool = False,
    ) -> Dict[str, Tensor]:
        batch_size = len(proposal_boxes)
        matched_gt_labels = cat(matched_gt_labels)
        matched_gt_boxes = cat(matched_gt_boxes)
        _proposal_boxes, batch_idx = cat_and_index(proposal_boxes)

        box_roi_features = self.box_pooler(
            features=features,
            proposal_boxes=_proposal_boxes,
            batch_idx=batch_idx,
            image_size=image_size,
        )  # [N, C, spatial]; N=num proposals passed, C=number of feature channels

        pred_detection = self.box_head[stage](box_roi_features)
        losses, _, _ = self.box_head[stage].compute_loss(
            prediction=pred_detection,
            target_labels=matched_gt_labels,
            matched_gt_boxes=matched_gt_boxes,
            proposals=_proposal_boxes,
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
        target_masks: Tensor,
        image_size: NDSIZE,
        num_instances: List[int],
        stage: int = 0,
        predict: bool = False,
    ) -> Dict[str, Tensor]:
        binary_masks = create_binary_masks(
            target_masks, num_instances=num_instances
        )  # List[Tensor]

        # compute mask loss on positive proposals
        pos_matched_gt_idx = []
        pos_proposal_boxes = []
        for gt_l, gt_idx, prop_b in zip(
            matched_gt_labels, matched_gt_idx, proposal_boxes
        ):
            pos_idx = torch.where(gt_l > 0)[0]
            pos_matched_gt_idx.append(gt_idx[pos_idx])
            pos_proposal_boxes.append(prop_b[pos_idx])

        target_masks_prepared = self.mask_pooler.pool_masks(
            binary_masks=binary_masks,
            proposal_boxes=pos_proposal_boxes,
            matched_gt_idx=pos_matched_gt_idx,
        )  # List[[R, output_size]]

        pos_proposal_boxes, batch_idx = cat_and_index(pos_proposal_boxes)

        mask_roi_features = self.mask_pooler(
            features=features,
            proposal_boxes=pos_proposal_boxes,
            batch_idx=batch_idx,
            image_size=image_size,
        )  # [N, C, spatial]; N=num proposals passed, C=number of feature channels

        pred_masks, _ = self.mask_head[stage](mask_roi_features)
        losses = self.mask_head[stage].compute_loss(
            pred_masks, torch.cat(target_masks_prepared, dim=0).unsqueeze(dim=1)
        )
        return losses, None

    def detach_proposals(
        self,
        proposals: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        return detach_all(proposals)

    @torch.no_grad()
    def assign_and_sample(
        self,
        proposals: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        targets: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        # optionally add proposals to prediction pool
        if self.gt_to_proposals:
            proposals = self.add_gt_to_proposals(proposals, targets)

        # match proposals to ground truth
        matched_gt_labels, matched_gt_boxes, matched_gt_idx = assign_targets_to_anchors(
            proposal_matcher=self.matcher[0],
            anchors=proposals["pred_boxes"],
            target_boxes=targets["target_boxes"],
            target_classes=targets["target_classes"],
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
                proposals["pred_boxes"][i] = cat(
                    [proposals["pred_boxes"][i], targets["target_boxes"][i]], dim=0
                )

                tc = targets["target_classes"][i]
                proposals["pred_labels"][i] = cat(
                    [proposals["pred_labels"][i], tc], dim=0
                )

                add_scores = torch.ones(
                    tc.shape[0],
                    device=proposals["pred_scores"][i].device,
                    dtype=proposals["pred_scores"][i].dtype,
                )
                proposals["pred_scores"][i] = cat(
                    [proposals["pred_scores"][i], add_scores], dim=0
                )
        return proposals

    def _inference_step_boxes(
        self,
        images: torch.Tensor,
        features: List[torch.Tensor],
        proposal_boxes: List[torch.Tensor],
        stage: int = 0,
    ):
        _proposal_boxes, batch_idx = cat_and_index(proposal_boxes)

        roi_features = self.box_pooler(
            features=features,
            proposal_boxes=_proposal_boxes,
            batch_idx=batch_idx,
            image_size=tuple(images.shape[2:]),
        )  # [P, C, spatial]

        pred_detection = self.box_head[stage](roi_features)

        image_shapes = [images.shape[2:]] * images.shape[0]
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
        return prediction

    def _inference_step_masks(
        self,
        images: torch.Tensor,
        features: List[torch.Tensor],
        proposal_boxes: List[torch.Tensor],
        stage: int = 0,
    ):
        _proposal_boxes, batch_idx = cat_and_index(proposal_boxes)
        roi_features = self.mask_pooler(
            features=features,
            proposal_boxes=_proposal_boxes,
            batch_idx=batch_idx,
            image_size=tuple(images.shape[2:]),
        )  # [P, C, spatial]

        pred_masks, _ = self.mask_head[stage](roi_features)
        # TODO: postprocessing
        # TODO: move cat and index to inference step

    # TODO: code duplication :/
    def postprocess_detections(
        self,
        pred_detection: Dict[str, torch.Tensor],
        proposal_boxes: List[torch.Tensor],
        image_shapes: List[Tuple[int]],
        stage: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposal_boxes]

        pred_detection = self.box_head[stage].postprocess_for_inference(
            pred_detection, proposal_boxes
        )
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
            if not self.box_head[stage].regress_multi_class:
                boxes, probs, labels = post_image_single_class_regression(
                    boxes=boxes,
                    probs=probs,
                    num_foreground_classes=self.num_foreground_classes,
                    image_shape=image_shape,
                    nms_thresh=self.roi_nms_thresh,
                    topk_candidates=None,
                    score_thresh=self.roi_score_thresh,
                    remove_small_boxes=None,
                    detections_per_img=self.roi_detections_per_img,
                )
            else:
                raise NotImplementedError

            all_boxes.append(boxes)
            all_probs.append(probs)
            all_labels.append(labels)
        return all_boxes, all_probs, all_labels


class RoIModule(BaseRoIModule):
    def train_step(
        self,
        images: torch.Tensor,
        features: List[torch.Tensor],
        proposals: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        targets: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        predict: bool = False,
    ):
        _features = [features[i] for i in self.decoder_levels]
        image_size = tuple(images.shape[2:])

        proposals = self.detach_proposals(proposals)
        (
            proposal_boxes,
            matched_gt_labels,
            matched_gt_boxes,
            matched_gt_idx,
        ) = self.assign_and_sample(proposals=proposals, targets=targets)

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
        if self.mask_mode_train:
            mask_losses, _ = self._train_step_masks(
                features=_features,
                matched_gt_labels=matched_gt_labels,
                matched_gt_idx=matched_gt_idx,
                proposal_boxes=proposal_boxes,
                target_masks=targets["target_masks"],
                image_size=image_size,
                num_instances=targets["target_num_instances"],
                stage=0,
                predict=False,
            )
            losses.update(mask_losses)
        return {f"roi_s0_{k}": i for k, i in losses.items()}, None

    @torch.no_grad()
    def inference_step(
        self,
        images: torch.Tensor,
        features: List[torch.Tensor],
        proposals: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        **kwargs,
    ) -> Dict[str, Any]:
        _features = [features[i] for i in self.decoder_levels]
        prediction = self._inference_step_boxes(
            images=images,
            features=_features,
            proposal_boxes=proposals["pred_boxes"],
        )

        if self.mask_mode_train:  # TODO: handle mask inference
            self._inference_step_masks(
                images=images,
                features=_features,
                proposal_boxes=proposals["pred_boxes"],
            )
        return prediction


RoIModuleType = TypeVar("RoIModuleType", bound=BaseRoIModule)
