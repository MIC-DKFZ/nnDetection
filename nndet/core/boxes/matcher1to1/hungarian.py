# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Original code from DETR https://github.com/facebookresearch/detr/blob/main/models/matcher.py
# SPDX-FileCopyrightText: 2020 Facebook
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from nndet.core.boxes.matcher1to1.base import BaseMatcher


class HungarianMatcher(BaseMatcher):
    # def match(self, outputs: Dict, targets: List[Dict]) -> List[Tuple[Tensor, Tensor]]:
    #     """Performs the matching
    #     Params:
    #         outputs: This is a dict that contains at least these entries:
    #              "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
    #              "pred_boxes": Tensor of dim [batch_size, num_queries, 6] with the predicted box coordinates
    #         targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
    #              "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
    #                        objects in the target) containing the class labels
    #              "boxes": Tensor of dim [num_target_boxes, 6] containing the target box coordinates
    #     Returns:
    #         A list of size batch_size, containing tuples of (index_i, index_j) where:
    #             - index_i is the indices of the selected predictions (in order)
    #             - index_j is the indices of the corresponding selected targets (in order)
    #         For each batch element, it holds:
    #             len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
    #     """

    @torch.no_grad()
    def match(
        self,
        pred_logits: torch.Tensor,
        pred_coords: torch.Tensor,
        target_boxes: List[torch.Tensor],
        target_labels: List[torch.Tensor],
    ) -> List[Tuple[Tensor, Tensor]]:
        """
        Perform matching over batch elements with at least one ground truth
        element in them

        Args:
            pred_logits: predicted class logits from model [B, R, C]
                where B=batch size, R=number of predictions, C=number of
                classes
            pred_coords: predicted bounding boxes coordinates from model
                [B, R, dims * 2] where B=batch size, R=number of predictions
                dims=number of spatial dimensions
            target_boxes: target ground truth boxes
                List([L, dims * 2]) where L is the number of ground truth boxes
                in each image, dims is the number of spatial dimensions and
                the length of the list corresponds to the batch size
            target_labels: target labels for each box List([L]) where L is the
                number of ground truth boxes in each image and
                the length of the list corresponds to the batch size

        Returns:
            List[Tuple[Tensor, Tensor]]: #TODO
        """
        bs, num_queries = pred_logits.shape[:2]

        # [batch_size * num_queries, num_classes]
        out_logits = pred_logits.flatten(0, 1)
        out_bbox = pred_coords.flatten(0, 1)  # [batch_size * num_queries, dims * 2]

        tgt_ids = torch.cat(target_labels, dim=0)
        tgt_bbox = torch.cat(target_boxes, dim=0)

        cost_class = sum(
            self.class_criterion[idx](out_logits, tgt_ids) for idx in range(len(self.class_criterion))
        )  # [batch_size * num_queries, num_gt_elements]
        cost_box = sum(
            self.box_criterion[idx](out_bbox, tgt_bbox) for idx in range(len(self.box_criterion))
        )  # [batch_size * num_queries, num_gt_elements]

        C = cost_class + cost_box
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v) for v in target_boxes]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
