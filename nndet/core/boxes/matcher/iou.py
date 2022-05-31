"""
Modifications licensed under

Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


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
