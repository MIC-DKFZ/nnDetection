# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence, Callable, Tuple, TypeVar
from abc import ABC

import torch
from torch import Tensor

from nndet.core.boxes.ops import box_iou


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
                 num_anchors_per_loc: int,
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
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
                        num_anchors_per_loc: int,
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
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

MatcherType = TypeVar('MatcherType', bound=Matcher)
