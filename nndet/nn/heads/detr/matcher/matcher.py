# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
from typing import Dict, List

import torch
from nndet.nn.heads.detr.matcher.matcher_funcs import FocalLossforMatcher, SimpleClassLossforMatcher
from nndet.core.boxes import box_cxcywhczd_to_xyxyzz
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from nndet.core.boxes import generalized_box_iou


class SimpleHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        **kwargs,
    ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.logits_to_probs = nn.Softmax(dim=-1)
        self.class_loss = SimpleClassLossforMatcher()
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    @staticmethod
    def fix_indices(mask: List[bool], indices: List[Tensor]):
        """
        :param mask: mask of which images in the batch contain boxes
        :param indices: List of indices from the matching
        :return: new index list which has the same length as the batch size
        """
        fixed_indices = []
        add_missing = 0
        for i, b in enumerate(mask):
            if b:
                # If the patch is non-empty, append the index from the matcher
                fixed_indices.append(indices[i - add_missing])
            else:
                # if the patch was empty, it was not used during the matching, so we have to add +1 to add_missing
                add_missing += 1
                # Append the "fake match" (0,0) to the indices list, because the ground truth is background for this, it
                # is fine
                fixed_indices.append(
                    (
                        torch.as_tensor([0], dtype=torch.int64),
                        torch.as_tensor([0], dtype=torch.int64),
                    )
                )
        return fixed_indices

    @torch.no_grad()
    def match(self, outputs: Dict, targets: List[Dict]):
        """Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 6] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 6] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        out_prob = self.logits_to_probs(outputs["pred_logits"]).flatten(
            0, 1
        )  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 6]
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        cost_class = self.class_loss(out_prob, tgt_ids)
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(
            box_cxcywhczd_to_xyxyzz(out_bbox),
            box_cxcywhczd_to_xyxyzz(tgt_bbox),
            eps=1e-8,
        )

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]

    def forward(self, outputs: Dict, targets: List[Dict]):

        # Filter out patches with no boxes in them
        num_boxes = 0
        mask = []
        masked_targets = []
        for i, t in enumerate(targets):
            if len(t["labels"]) == 0:
                mask.append(False)
            else:
                num_boxes += len(t["labels"])
                masked_targets.append(t)
                mask.append(True)
        # use filtered outputs for matching
        masked_outputs = {
            "pred_logits": outputs["pred_logits"][mask],
            "pred_boxes": outputs["pred_boxes"][mask],
        }

        # matching
        masked_indices = None
        if num_boxes > 0:
            masked_indices = self.match(masked_outputs, masked_targets)

        full_indices = self.fix_indices(mask, masked_indices)
        return (
            num_boxes,
            mask,
            masked_indices,
            full_indices,
            masked_outputs,
            masked_targets,
        )


class FocalHungarianMatcher(SimpleHungarianMatcher):
    """
    Matcher Class similar to SimpleHungarianMatcher with focal loss as class loss and sigmoid as logits conversion
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        alpha: float = 0.75,
        gamma: float = 1,
        dino: bool = False,
        **kwargs,
    ):
        super().__init__(cost_class, cost_bbox, cost_giou, **kwargs)
        self.logits_to_probs = nn.Sigmoid()
        self.class_loss = FocalLossforMatcher(alpha=alpha, gamma=gamma)
        self.dino = dino
