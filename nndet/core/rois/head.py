# """
# For each stage:
#     (images: Tensor, features: List[Tensor], proposals: dict, gt: dict)
#     match proposals to gt
#     subsample
#     predict features
#     compute loss
# """
from typing import TypeVar, List, Dict, Union, Tuple, Sequence, Any

import torch

from nndet.arch.heads.comb import RoIHeadType
from nndet.core.boxes import MatcherType
from nndet.core.rois.pooler import PoolerType
from nndet.core.boxes.sampler import SamplerType

from nndet.core.boxes.utils import cat_and_index, extend_and_cat_boxes
from nndet.core.boxes.assign import assign_targets_to_anchors
from nndet.core.boxes.post import post_image_single_class_regression


# TODO: refactor module name
class RoIModule(torch.nn.Module):
    def __init__(self,
                 box_head: RoIHeadType, # use head without sampler
                 matcher: MatcherType,
                 pooler: PoolerType,
                 sampler: SamplerType, # NegativeSampler default => random balanced sampling
                 num_classes: int,
                 decoder_levels: Sequence[int],
                 gt_to_proposals: bool = True,
                  # post-processing
                 roi_score_thresh: float = None,
                 roi_detections_per_img: int = 100,
                 roi_nms_thresh: float = 0.9,
                 ) -> None:
        super().__init__()
        self.box_head = box_head
        self.matcher = matcher
        self.pooler = pooler
        self.sampler = sampler
        self.decoder_levels = decoder_levels
        self.gt_to_proposals = gt_to_proposals

        self.num_foreground_classes = num_classes
        self.roi_score_thresh = roi_score_thresh
        self.roi_detections_per_img = roi_detections_per_img
        self.roi_nms_thresh = roi_nms_thresh

    def train_step(self,
                   images: torch.Tensor,
                   features: List[torch.Tensor],
                   proposals: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
                   targets: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
                   predict: bool = False,
                   ):
        _features = [features[i] for i in self.decoder_levels]
        
        losses, prediction, roi_features, inds, pos_inds, neg_inds = self._train_step_boxes(
                features=_features,
                proposal_boxes=proposals["pred_boxes"],
                proposal_scores=proposals["pred_scores"],
                target_boxes=targets["target_boxes"],
                target_classes=targets["target_classes"],
                image_size=tuple(images.shape[2:]),
                predict=predict,
            )
        return losses, prediction

    def _train_step_boxes(
        self,
        features: List[torch.Tensor],
        proposal_boxes: List[torch.Tensor],
        proposal_scores: List[torch.Tensor],
        target_boxes: List[torch.Tensor],
        target_classes: List[torch.Tensor],
        image_size: Union[Tuple[int, int], Tuple[int, int, int]],
        predict: bool = False,
        ):
        if self.gt_to_proposals:
            proposal_boxes = self.add_gt_to_proposals(proposal_boxes, target_boxes)

        inds, pos_inds, neg_inds, labels, matched_gt_boxes = self.sample_and_match(
            proposal_boxes,
            proposal_scores,
            target_boxes,
            target_classes,
        )
        proposal_boxes, batch_idx = cat_and_index(proposal_boxes)
        proposal_boxes = proposal_boxes[inds]
        batch_idx = batch_idx[inds]

        roi_features = self.pooler(
            features=features,
            proposal_boxes=proposal_boxes,
            batch_idx=batch_idx,
            image_size=image_size,
            ) # [P, C, spatial]
        pred_detection = self.box_head(roi_features)

        losses, _, _ = self.box_head.compute_loss(
            pred_detection, labels, matched_gt_boxes, proposal_boxes)

        prediction = None
        if predict:
            raise NotImplementedError
        return losses, prediction, roi_features, inds, pos_inds, neg_inds

    @torch.no_grad()
    def inference_step(self,
                       images: torch.Tensor,
                       features: List[torch.Tensor],
                       proposals: Dict[str, Union[torch.Tensor,
                                                  List[torch.Tensor]]],
                       **kwargs,
                       ) -> Dict[str, Any]:
        _features = [features[i] for i in self.decoder_levels]
        prediction = self._inference_step_boxes(
            images=images,
            features=_features,
            proposal_boxes=proposals["pred_boxes"],
        )
        return prediction

    def _inference_step_boxes(self,
                              images: torch.Tensor,
                              features: List[torch.Tensor],
                              proposal_boxes: List[torch.Tensor],
                              ):
        _proposal_boxes, batch_idx = cat_and_index(proposal_boxes)

        roi_features = self.pooler(
            features=features,
            proposal_boxes=_proposal_boxes,
            batch_idx=batch_idx,
            image_size=tuple(images.shape[2:])
            ) # [P, C, spatial]
        pred_detection = self.box_head(roi_features)

        image_shapes = [images.shape[2:]] * images.shape[0]
        boxes, probs, labels = self.postprocess_detections(
            pred_detection=pred_detection,
            proposal_boxes=proposal_boxes,
            image_shapes=image_shapes,
        )
        prediction = {
            "pred_boxes": boxes,
            "pred_scores": probs,
            "pred_labels": labels,
            }
        return prediction

    # TODO: code duplication :/
    def postprocess_detections(
        self,
        pred_detection: Dict[str, torch.Tensor],
        proposal_boxes: List[torch.Tensor],
        image_shapes: List[Tuple[int]],
        ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposal_boxes]
        
        pred_detection = self.box_head.postprocess_for_inference(pred_detection, proposal_boxes)
        pred_boxes, pred_probs = pred_detection["pred_boxes"], pred_detection["pred_probs"]

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_probs = pred_probs.split(boxes_per_image, 0)

        all_boxes, all_probs, all_labels = [], [], []
        # iterate over images
        for boxes, probs, image_shape in zip(pred_boxes, pred_probs, image_shapes):
            if not self.box_head.regress_multi_class:
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

    @torch.no_grad()
    def sample_and_match(self,
                         proposal_boxes: List[torch.Tensor],
                         proposal_scores: List[torch.Tensor],
                         target_boxes: List[torch.Tensor],
                         target_classes: List[torch.Tensor],
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create target labels and sample RoIs for further processing

        Args:
            proposal_boxes: proposed bounding boxes
                (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            proposal_scores: predicted score for each bounding box [N]
            target_boxes: ground truth bounding boxes
                (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            target_classes: ground truth class for each box [N]

        Returns:
            Tensor: concatenated and extended proposals with batch index
                (batch_idx, x1, y1, x2, y2, (z1, z2))[N, 1 + dim * 2]
            labels: matched label for each proposal [N] (
                [1, K]: foreground classes, 0: background, -1: between)
            Tensor: matched gt box [N, dim * 2]
        """
        # match proposals to ground truth
        labels, matched_gt_boxes = assign_targets_to_anchors(
            proposal_matcher=self.matcher,
            anchors=proposal_boxes,
            target_boxes=target_boxes,
            target_classes=target_classes,
            )

        # subsample rois
        pos_mask, neg_mask = self.sampler(
            target_labels=labels,
            fg_probs=proposal_scores,
            )
        sampled_pos_inds = torch.where(torch.cat(pos_mask, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(neg_mask, dim=0))[0]
        inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        _labels = torch.cat(labels, dim=0)[inds]
        _matched_gt_boxes = torch.cat(matched_gt_boxes, dim=0)[inds]
        
        return inds, sampled_pos_inds, sampled_neg_inds, _labels, _matched_gt_boxes

    def add_gt_to_proposals(self,
                            proposals: List[torch.Tensor],
                            gt: List[torch.Tensor],
                            ) -> List[torch.Tensor]:
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
        return [torch.cat([p, g], dim=0) for p, g in zip(proposals, gt)]


RoIModuleType = TypeVar('RoIModuleType', bound=RoIModule)


class CascadeRoIModule(RoIModule):
    """
    add gt [first stage] -> forward all RoIs -> subsample for los
    """
    pass


class Sequencer(torch.nn.Module):
    def __init__(self,
                 roi_heads: List[RoIModuleType],
                 ) -> None:
        """
        Cascade multiple RoI Heads
        
        TODO: gradient scaling
        TODO: detach proposals
        """
        super().__init__()
        self.roi_heads = torch.nn.ModuleList(roi_heads)

    def forward(self):
        for head in self.roi_heads:
            # predict rois
            pass
        pass
