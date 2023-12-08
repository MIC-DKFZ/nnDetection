# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Original code from DETR https://github.com/facebookresearch/detr/blob/main/models/matcher.py
# SPDX-FileCopyrightText: 2020 Facebook, Inc
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Tuple

import torch
from scipy.optimize import linear_sum_assignment

from nndet.core.boxes.matcher1to1.base import BaseMatcher


class HungarianMatcher(BaseMatcher):
    @torch.no_grad()
    def match(
        self,
        pred_logits: torch.Tensor,
        pred_coords: torch.Tensor,
        target_boxes: List[torch.Tensor],
        target_labels: List[torch.Tensor],
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
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
            Optional[Dict]: Dict containing the matching cost
        """
        bs, num_queries = pred_logits.shape[:2]

        # [batch_size * num_queries, num_classes]
        out_logits = pred_logits.flatten(0, 1)
        out_bbox = pred_coords.flatten(0, 1)  # [batch_size * num_queries, dims * 2]

        tgt_labels = torch.cat(target_labels, dim=0)
        num_boxes = tgt_labels.shape[0]
        tgt_bbox = torch.cat(target_boxes, dim=0)
        num_class_criterion = len(self.class_criterion)
        num_box_criterion = len(self.box_criterion)
        cost_classes = [self.class_criterion[idx](out_logits, tgt_labels) for idx in range(num_class_criterion)]
        cost_boxes = [self.box_criterion[idx](out_bbox, tgt_bbox) for idx in range(num_box_criterion)]
        cost_class = sum(cost_classes)  # [batch_size * num_queries, num_gt_elements]
        cost_box = sum(cost_boxes)  # [batch_size * num_queries, num_gt_elements]

        C = cost_class + cost_box
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v) for v in target_boxes]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        out_indices = [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
        crit_log_dict = None
        if self.extended_logging:
            crit_log_dict = self._get_log_dict(
                bs=bs,
                num_queries=num_queries,
                sizes=sizes,
                out_indices=out_indices,
                cost_classes=cost_classes,
                cost_boxes=cost_boxes,
                num_boxes=num_boxes,
            )
        return out_indices, crit_log_dict

    def _get_log_dict(
        self,
        bs: int,
        num_queries: int,
        sizes: List[int],
        out_indices: List[Tuple[torch.Tensor, torch.Tensor]],
        cost_classes: List[torch.Tensor],
        cost_boxes: List[torch.Tensor],
        num_boxes: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Provide additional information for logging from criterion values

        Args:
            bs: batch size
            num_queries: number of queries (aka number of predictions)
            sizes: number of ground truth boxes per image
            out_indices: indices as determined by the matching algorithm
            cost_classes: costs of class criterions
            cost_boxes: costs of box criterions
            num_boxes: number of ground truth boxes

        Returns:
            Dict[str, torch.Tensor]: additional information for logging
        """
        num_class_criterion = len(self.class_criterion)
        num_box_criterion = len(self.box_criterion)

        crit_log_dict = {}
        # Initialize average keys (normalized by number of boxes)
        for j in range(num_class_criterion):
            crit_log_dict[f"__class_crit_{j}_avg"] = 0
        for j in range(num_box_criterion):
            crit_log_dict[f"__box_crit_{j}_avg"] = 0

        for i, (pred_indices, gt_indices) in enumerate(out_indices):
            # class costs
            for j in range(num_class_criterion):
                cost_classes_tmp = (
                    cost_classes[j].view(bs, num_queries, -1).split(sizes, -1)[i][i][pred_indices, gt_indices]
                )
                crit_log_dict[f"__class_crit_{j}_avg"] += cost_classes_tmp.sum() / num_boxes
                for k, cost_class_tmp in enumerate(cost_classes_tmp):
                    crit_log_dict[f"__class_crit_{j}_img_{i}_box_{k}"] = cost_class_tmp

            # reg costs
            for j in range(num_box_criterion):
                cost_boxes_tmp = (
                    cost_boxes[j].view(bs, num_queries, -1).split(sizes, -1)[i][i][pred_indices, gt_indices]
                )
                crit_log_dict[f"__box_crit_{j}_avg"] += cost_boxes_tmp.sum() / num_boxes
                for k, cost_box_tmp in enumerate(cost_boxes_tmp):
                    crit_log_dict[f"__box_crit_{j}_img_{i}_box_{k}"] = cost_box_tmp

        # Get total average
        crit_log_dict["__crit_avg"] = sum([value if "avg" in key else 0 for key, value in crit_log_dict.items()])
        return crit_log_dict
