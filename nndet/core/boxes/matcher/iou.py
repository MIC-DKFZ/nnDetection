# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from torchvision (https://github.com/pytorch/vision) licensed under
# SPDX-FileCopyrightText: 2016 Soumith Chintala
# SPDX-License-Identifier: BSD-3-Clause


from typing import Callable, Tuple

import torch
from torch import Tensor
from loguru import logger

from nndet.core.boxes.ops import box_iou
from nndet.core.boxes.matcher.base import Matcher


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
