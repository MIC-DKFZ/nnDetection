# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import torch
from loguru import logger
from torch import Tensor

from nndet.core.boxes import MatcherType
from nndet.core.boxes.assign import assign_targets_to_anchors
from nndet.core.boxes.ops import cat_and_index
from nndet.core.boxes.sampler import SamplerType
from nndet.core.post.box import BoxPostprocessing
from nndet.core.post.mask import MaskPostprocessing
from nndet.core.rois.pooler import RoIPooler
from nndet.nn.heads.comb.roi import RoIHead
from nndet.nn.heads.masker.base import Masker
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
        matcher: Union[MatcherType, List[MatcherType], Tuple[MatcherType]],
        sampler: SamplerType,  # NegativeSampler default => random balanced sampling
        num_classes: int,
        decoder_levels: Sequence[int],
        gt_to_proposals: bool = True,
        # mask
        mask_head: Optional[Union[Masker, List[Masker], Tuple[Masker]]] = None,
        mask_pooler: Optional[RoIPooler] = None,
        mask_post: Optional[MaskPostprocessing] = None,
        # post-processing
        roi_score_thresh: float = None,
        roi_detections_per_img: int = 100,
        roi_nms_thresh: float = 0.6,
        inference_prob_rpn: bool = False,
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

        self.roi_score_thresh = roi_score_thresh
        self.roi_detections_per_img = roi_detections_per_img
        self.roi_nms_thresh = roi_nms_thresh
        self.inference_prob_rpn = inference_prob_rpn

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
        image_size: ND_TUPLE_INT,
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
        target_binary_masks: Tensor,
        image_size: ND_TUPLE_INT,
        stage: int = 0,
        predict: bool = False,
    ) -> Dict[str, Tensor]:
        # compute mask loss on positive proposals
        pos_matched_gt_idx = []
        pos_proposal_boxes = []
        for gt_l, gt_idx, prop_b in zip(matched_gt_labels, matched_gt_idx, proposal_boxes):
            pos_idx = torch.where(gt_l > 0)[0]
            pos_matched_gt_idx.append(gt_idx[pos_idx])
            pos_proposal_boxes.append(prop_b[pos_idx])

        target_masks_prepared = self.mask_pooler.pool_masks(
            binary_masks=target_binary_masks,
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
        target_masks_prepared_batched = torch.cat(target_masks_prepared, dim=0).unsqueeze(dim=1)
        assert pred_masks.shape[0] == target_masks_prepared_batched.shape[0]
        losses = self.mask_head[stage].compute_loss(
            pred_masks,
            target_masks_prepared_batched,
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
        images: torch.Tensor,
        features: List[torch.Tensor],
        proposal_boxes: List[torch.Tensor],
        proposal_scores: Optional[List[torch.Tensor]] = None,
        stage: int = 0,
    ) -> Dict[str, List[torch.Tensor]]:
        _proposal_boxes, batch_idx = cat_and_index(proposal_boxes)

        if _proposal_boxes.numel() == 0:
            batch_size = len(proposal_boxes)
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
                image_size=tuple(images.shape[2:]),
            )  # [P, C, spatial]

            pred_detection = self.box_head[stage](roi_features)

            image_shapes = [images.shape[2:]] * images.shape[0]
            boxes, probs, labels = self.postprocess_detections(
                pred_detection=pred_detection,
                proposal_boxes=proposal_boxes,
                image_shapes=image_shapes,
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
        images: torch.Tensor,
        features: List[torch.Tensor],
        pred_boxes: List[torch.Tensor],
        pred_probs: List[torch.Tensor],
        pred_labels: List[torch.Tensor],
        stage: int = 0,
    ) -> Dict[str, List[Tensor]]:
        _boxes, batch_idx = cat_and_index(pred_boxes)

        if _boxes.numel() == 0:
            batch_size = len(pred_boxes)
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
                image_size=tuple(images.shape[2:]),
            )  # [P, C, spatial]

            _masks, _ = self.mask_head[stage](roi_features)  # [P, C, mask_dim]

            image_shapes = [images.shape[2:]] * images.shape[0]
            masks, probs, labels = self.postprocess_masks(
                masks=_masks,
                pred_probs=pred_probs,
                pred_labels=pred_labels,
                image_shapes=image_shapes,
                stage=stage,
            )
        prediction = {
            "pred_masks": masks,
            "pred_mask_scores": probs,
            "pred_mask_labels": labels,
            "__pred_image_spatial_size": tuple(images.shape[2:]),
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
        masks_per_image = [len(pl) for pl in pred_labels]
        assert [len(pp) == len(pl) for pp, pl in zip(pred_probs, pred_labels)]
        assert sum(masks_per_image) == masks.shape[0]

        pred_masks = self.mask_head[stage].logits_to_probs(masks, pred_labels)
        pred_masks = pred_masks.split(masks_per_image, 0)

        return self.mask_post.process_batch(
            reps=pred_masks,
            probs=pred_probs,
            labels=pred_labels,
        )


class RoIModule(BaseRoIModule):
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
                target_binary_masks=targets["target_binary_masks"],
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

                ``"__pred_image_spatial_size"`` ND_TUPLE_INT
                    image size which was used for prediction. Needed to restore
                    correct size of image when pasting binary masks.

        """
        _features = [features[i] for i in self.decoder_levels]
        prediction = self._inference_step_boxes(
            images=images,
            features=_features,
            proposal_boxes=proposals["pred_boxes"],
            proposal_scores=proposals["pred_scores"],
        )

        if self.mask_mode:
            mask_preds = self._inference_step_masks(
                images=images,
                features=_features,
                pred_boxes=prediction["pred_boxes"],
                pred_probs=prediction["pred_scores"],
                pred_labels=prediction["pred_labels"],
            )
            prediction.update(mask_preds)
        return prediction


RoIModuleType = TypeVar("RoIModuleType", bound=BaseRoIModule)
