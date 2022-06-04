# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from mmdetection licensed under
# SPDX-FileCopyrightText: 2018-2023 OpenMMLab
# SPDX-License-Identifier: Apache-2.0


from typing import Sequence, Callable, Tuple

import torch
from torch import Tensor
from loguru import logger

from nndet.core.boxes.ops import box_iou, box_center_dist, center_in_boxes
from nndet.core.boxes.matcher.base import Matcher

INF = 100  # not really inv but here it is sufficient


class ATSSMatcher(Matcher):
    def __init__(self,
                 num_candidates: int,
                 similarity_fn: Callable[[Tensor, Tensor], Tensor] = box_iou,
                 center_in_gt: bool = True,
                 ):
        """
        Compute matching based on ATSS
        https://arxiv.org/abs/1912.02424
        `Bridging the Gap Between Anchor-based and Anchor-free Detection
        via Adaptive Training Sample Selection`

        Args:
            num_candidates: number of positions to select candidates from
            similarity_fn: function for similarity computation between
                boxes and anchors
            center_in_gt: If diabled, matched anchor center points do not need
                to lie withing the ground truth box.
        """
        super().__init__(similarity_fn=similarity_fn)
        self.num_candidates = num_candidates
        self.min_dist = 0.01
        self.center_in_gt = center_in_gt
        logger.info(f"Running ATSS Matching with num_candidates={self.num_candidates} "
                    f"and center_in_gt {self.center_in_gt}.")

    def compute_matches(self,
                        boxes: torch.Tensor,
                        anchors: torch.Tensor,
                        num_anchors_per_level: Sequence[int],
                        num_anchors_per_loc: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute matches according to ATTS for a single image
        Adapted from
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/assigners/atss_assigner.py
        https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py

        Args:
            boxes: anchors are matches to these boxes (e.g. ground truth)
                [N, dims * 2](x1, y1, x2, y2, (z1, z2))
            anchors: anchors to match [M, dims * 2](x1, y1, x2, y2, (z1, z2))
            num_anchors_per_level: number of anchors per feature pyramid level
            num_anchors_per_loc: number of anchors per position

        Returns:
            Tensor: matrix which contains the similarity from each boxes
                to each anchor [N, M]
            Tensor: vector which contains the matched box index for all
                anchors (if background `BELOW_LOW_THRESHOLD` is used
                and if it should be ignored `BETWEEN_THRESHOLDS` is used)
                [M]
        """
        num_gt = boxes.shape[0]
        num_anchors = anchors.shape[0]

        distances, _, anchors_center = box_center_dist(boxes, anchors)  # num_boxes x anchors

        # select candidates based on center distance
        candidate_idx = []
        start_idx = 0
        for level, apl in enumerate(num_anchors_per_level):
            end_idx = start_idx + apl

            selectable_k = min(self.num_candidates * num_anchors_per_loc, apl)
            _, idx = distances[:, start_idx: end_idx].topk(selectable_k, dim=1, largest=False)
            # idx shape [num_boxes x selectable_k]
            candidate_idx.append(idx + start_idx)

            start_idx = end_idx
        # [num_boxes x num_candidates] (index of candidate anchors)
        candidate_idx = torch.cat(candidate_idx, dim=1)

        match_quality_matrix = self.similarity_fn(boxes, anchors)  # [num_boxes x anchors]
        candidate_overlaps = match_quality_matrix.gather(1, candidate_idx)  # [num_boxes, n_candidates]

        # compute adaptive iou threshold
        overlaps_mean_per_gt = candidate_overlaps.mean(dim=1)  # [num_boxes]
        overlaps_std_per_gt = candidate_overlaps.std(dim=1)  # [num_boxes]
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt  # [num_boxes]
        is_pos = candidate_overlaps >= overlaps_thr_per_gt[:, None]  # [num_boxes x n_candidates]

        if self.center_in_gt:  # can discard all candidates in case of very small objects :/
            # center point of selected anchors needs to lie within the ground truth
            boxes_idx = torch.arange(num_gt, device=boxes.device, dtype=torch.long)[:, None]\
                .expand_as(candidate_idx).contiguous()  # [num_boxes x n_candidates]
            is_in_gt = center_in_boxes(
                anchors_center[candidate_idx.view(-1)], boxes[boxes_idx.view(-1)], eps=self.min_dist)
            is_pos = is_pos & is_in_gt.view_as(is_pos)  # [num_boxes x n_candidates]

        # in case on anchor is assigned to multiple boxes, use box with highest IoU
        for ng in range(num_gt):
            candidate_idx[ng, :] += ng * num_anchors
        overlaps_inf = torch.full_like(match_quality_matrix, -INF).view(-1)
        index = candidate_idx.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = match_quality_matrix.view(-1)[index]
        overlaps_inf = overlaps_inf.view_as(match_quality_matrix)

        matched_vals, matches = overlaps_inf.max(dim=0)
        matches[matched_vals == -INF] = self.BELOW_LOW_THRESHOLD
        # print(f"Num matches {(matches >= 0).sum()}, Adapt IoU {overlaps_thr_per_gt}")
        return match_quality_matrix, matches
