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
            List[Tuple[Tensor, Tensor]]: returns the matched indices for a
                batch. The first tensor contains the selected predictions
                (in order) and the second tensor contains the selected ground
                truth objects (in order). It holds for each elements:
                len(index_i) = len(index_j) = min(num_pred, num_target_boxes)
        """
        bs, num_queries = pred_logits.shape[:2]

        # [batch_size * num_queries, num_classes]
        out_logits = pred_logits.flatten(0, 1)
        out_bbox = pred_coords.flatten(0, 1)  # [batch_size * num_queries, dims * 2]

        tgt_labels = torch.cat(target_labels, dim=0)
        tgt_bbox = torch.cat(target_boxes, dim=0)

        cost_class = sum(
            self.class_criterion[idx](out_logits, tgt_labels) for idx in range(len(self.class_criterion))
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
