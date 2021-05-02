"""
For each stage:
    (images: Tensor, features: List[Tensor], proposals: dict, gt: dict)
    match proposals to gt
    subsample
    predict features
    compute loss
"""
from typing import TypeVar, List, Dict, Union, Tuple

import torch

from nndet.arch.heads.comb import HeadType
from nndet.core.boxes import MatcherType
from nndet.core.rois.pooler import PoolerType
from nndet.core.boxes.sampler import SamplerType

from nndet.core.boxes.utils import extend_and_cat_boxes
from nndet.core.boxes.assign import assign_targets_to_anchors


class RoIHead(torch.nn.Module):
    def __init__(self,
                 box_head: HeadType, # use head without sampler
                 matcher: MatcherType,
                 pooler: PoolerType,
                 sampler: SamplerType, # NegativeSampler default => random balanced sampling
                 gt_to_proposals: bool = True,
                 ) -> None:
        super().__init__()
        self.box_head = box_head
        self.matcher = matcher
        self.pooler = pooler
        self.sampler = sampler
        self.gt_to_proposals = gt_to_proposals

    def forward(self, roi_features: torch.Tensor):
        # TODO: correct spatial dims for processing
        return self.box_head([roi_features])

    def train_step(self,
                   images: torch.Tensor,
                   features: List[torch.Tensor],
                   proposals: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
                   targets: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
                   ):
        pass

    def _train_step_boxes(
        self,
        features: List[torch.Tensor],
        proposals: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        targets: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    ):
        target_boxes: List[torch.Tensor] = targets["target_boxes"]
        target_classes: List[torch.Tensor] = targets["target_classes"]        

        proposal_boxes = proposals["pred_boxes"]
        proposal_scores = proposals["pred_scores"]
 
        if self.gt_to_proposals:
            proposal_boxes = self.add_gt_to_proposals(proposal_boxes, target_boxes)

        proposal_boxes_sampled, labels, matched_gt_boxes = self.sample_and_match(
            proposal_boxes,
            proposal_scores,
            target_boxes,
            target_classes,
        )

        roi_features = self.pooler(features, proposal_boxes_sampled) # [P, C, spatial]
        pred_detection = self(roi_features)

        # compute loss
        losses, pos_idx, neg_idx = self.box_head.compute_loss(
            pred_detection, labels, matched_gt_boxes, proposal_boxes_sampled)
        return losses

    @torch.no_grad
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
            proposal_boxes, target_boxes, target_classes,
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
        _proposal_boxes = extend_and_cat_boxes(proposal_boxes)[inds]
        return _proposal_boxes, _labels, _matched_gt_boxes

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


RoIHeadType = TypeVar('RoIHeadType', bound=RoIHead)


class Sequencer(torch.nn.Module):
    def __init__(self,
                 roi_heads: List[RoIHeadType],
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
