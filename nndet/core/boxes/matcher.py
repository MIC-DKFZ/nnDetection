"""
Parts of this code are from torchvision and thus licensed under

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016, 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

from typing import Sequence, Callable, Tuple, TypeVar
from abc import ABC

import torch
from torch import Tensor
from loguru import logger

from nndet.core.boxes.ops import box_iou, box_center_dist, center_in_boxes

INF = 100  # not really inv but here it is sufficient


class Matcher(ABC):
    BELOW_LOW_THRESHOLD: int = -1
    BETWEEN_THRESHOLDS: int = -2

    def __init__(self, similarity_fn: Callable[[Tensor, Tensor], Tensor] = box_iou):
        """
        Matches boxes and anchors to each other

        Args:
            similarity_fn: function for similarity computation between
                boxes and anchors
        """
        self.similarity_fn = similarity_fn

    def __call__(self,
                 boxes: torch.Tensor,
                 anchors: torch.Tensor,
                 num_anchors_per_level: Sequence[int],
                 num_anchors_per_loc: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute matches for a single image

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
        if boxes.numel() == 0:
            # no ground truth
            num_anchors = anchors.shape[0]
            match_quality_matrix = torch.tensor([]).to(anchors)
            matches = torch.empty(num_anchors, dtype=torch.int64).fill_(self.BELOW_LOW_THRESHOLD)
            return match_quality_matrix, matches
        else:
            # at least one ground truth
            return self.compute_matches(
                boxes=boxes, anchors=anchors,
                num_anchors_per_level=num_anchors_per_level,
                num_anchors_per_loc=num_anchors_per_loc,
                )

    def compute_matches(self,
                        boxes: torch.Tensor,
                        anchors: torch.Tensor,
                        num_anchors_per_level: Sequence[int],
                        num_anchors_per_loc: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute matches

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
        raise NotImplementedError


class IoUMatcher(Matcher):
    def __init__(self,
                 low_threshold: float,
                 high_threshold: float,
                 allow_low_quality_matches: bool,
                 similarity_fn: Callable[[Tensor, Tensor], Tensor] = box_iou):
        """
        Compute IoU based matching for a single image

        Args:
            low_threshold: threshold used to assign background values
            high_threshold: threshold used to assign foreground values
            allow_low_quality_matches: if enabled, anchors with not
                match get the box with highest IoU assigned
            similarity_fn: function for similarity computation between
                boxes and anchors
        """
        super().__init__(similarity_fn=similarity_fn)
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def compute_matches(self,
                        boxes: torch.Tensor,
                        anchors: torch.Tensor,
                        **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute matches according to given iou thresholds
        Adapted from
        (https://github.com/pytorch/vision/blob/c7c2085ec686ccc55e1df85736b240b24
        05d1179/torchvision/models/detection/_utils.py)

        Args:
            boxes: anchors are matches to these boxes (e.g. ground truth)
                [N, dims * 2](x1, y1, x2, y2, (z1, z2))
            anchors: anchors to match [M, dims * 2](x1, y1, x2, y2, (z1, z2))
            anchors_per_level: number of anchors per feature pyramid level
            anchors_per_loc: number of anchors per position

        Returns:
            Tensor: matrix which contains the similarity from each boxes
                to each anchor [N, M]
            Tensor: vector which contains the matched box index for all
                anchors (if background `BELOW_LOW_THRESHOLD` is used
                and if it should be ignored `BETWEEN_THRESHOLDS` is used)
                [M]
        """
        match_quality_matrix = self.similarity_fn(boxes, anchors)

        # match_quality_matrix is M (gt) x N (anchors)
        # Max over gt elements (dim 0) to find best gt candidate for each anchor
        matched_vals, matches = match_quality_matrix.max(dim=0)
        # _v, _i = matched_vals.topk(5)
        # print(boxes, _v, anchors[_i])
        if self.allow_low_quality_matches:
            all_matches = matches.clone()

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
                matched_vals < self.high_threshold
        )
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            matches = self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        # self._debug_logging(match_quality_matrix, matches, matched_vals,
        #                     below_low_threshold, between_thresholds)

        return match_quality_matrix, matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Find the best matching prediction for each bounding box
        regardless of its IoU (this implementation excludes ties!)

        Args:
            matches: matched anchors to background and in between
            all_matches: all matches regardless of IoU
            match_quality_matrix: [M,N] tensor of IoUs (GroundTruth x NumAnchors)
        """
        # For each gt, find the prediction with has highest quality
        _, best_pred_idx = match_quality_matrix.max(dim=1)  # [M]
        matches[best_pred_idx] = torch.arange(len(best_pred_idx)).to(matches)
        return matches

    @staticmethod
    def _debug_logging(match_quality_matrix, matches, matched_vals,
                       below_low_threshold, between_thresholds):
        logger.info("########## Matcher ##############")
        logger.info(f"Max IoU: {match_quality_matrix.max()}")
        logger.info(f"Foreground IoUs: {matched_vals[matches > -1]}")
        logger.info(f"Num GT: {match_quality_matrix.shape[0]}")
        match_bet_min = matched_vals[between_thresholds].min() if \
            matched_vals[between_thresholds].nelement() > 0 else None
        match_bet_max = matched_vals[between_thresholds].max() if \
            matched_vals[between_thresholds].nelement() > 0 else None
        logger.info(f"Inbetween IoU ranging from {match_bet_min} to {match_bet_max}")
        logger.info(f"Max background IoU: {matched_vals[below_low_threshold].max()}")
        logger.info("#################################")


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
        (https://github.com/sfzhang15/ATSS/blob/79dfb28bd1/atss_core/modeling/rpn/atss
        /loss.py#L180-L184)

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

        distances, boxes_center, anchors_center = box_center_dist(boxes, anchors)  # num_boxes x anchors

        # select candidates based on center distance
        candidate_idx = []
        start_idx = 0
        for level, apl in enumerate(num_anchors_per_level):
            end_idx = start_idx + apl

            topk = min(self.num_candidates * num_anchors_per_loc, apl)
            _, idx = distances[:, start_idx: end_idx].topk(topk, dim=1, largest=False)
            # idx shape [num_boxes x topk]
            candidate_idx.append(idx + start_idx)

            start_idx = end_idx
        # [num_boxes x num_candidates] (index of candidate anchors)
        candidate_idx = torch.cat(candidate_idx, dim=1)

        match_quality_matrix = self.similarity_fn(boxes, anchors)  # [num_boxes x anchors]
        candidate_ious = match_quality_matrix.gather(1, candidate_idx)  # [num_boxes, n_candidates]

        # compute adaptive iou threshold
        iou_mean_per_gt = candidate_ious.mean(dim=1)  # [num_boxes]
        iou_std_per_gt = candidate_ious.std(dim=1)  # [num_boxes]
        iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt  # [num_boxes]
        is_pos = candidate_ious >= iou_thresh_per_gt[:, None]  # [num_boxes x n_candidates]

        if self.center_in_gt:  # can discard all candidates in case of very small objects :/
            # center point of selected anchors needs to lie within the ground truth
            boxes_idx = torch.arange(num_gt, device=boxes.device, dtype=torch.long)[:, None]\
                .expand_as(candidate_idx).contiguous()  # [num_boxes x n_candidates]
            is_in_gt = center_in_boxes(
                anchors_center[candidate_idx.view(-1)], boxes[boxes_idx.view(-1)], eps=self.min_dist)
            is_pos = is_pos & is_in_gt.view_as(is_pos)  # [num_boxes x n_candidates]

        # in case on anchor is assigned to multiple boxes, use box with highest IoU
        # TODO: think about a better way to do this
        for ng in range(num_gt):
            candidate_idx[ng, :] += ng * num_anchors
        ious_inf = torch.full_like(match_quality_matrix, -INF).view(-1)
        index = candidate_idx.view(-1)[is_pos.view(-1)]
        ious_inf[index] = match_quality_matrix.view(-1)[index]
        ious_inf = ious_inf.view_as(match_quality_matrix)

        matched_vals, matches = ious_inf.max(dim=0)
        matches[matched_vals == -INF] = self.BELOW_LOW_THRESHOLD
        # print(f"Num matches {(matches >= 0).sum()}, Adapt IoU {iou_thresh_per_gt}")
        return match_quality_matrix, matches


MatcherType = TypeVar('MatcherType', bound=Matcher)
