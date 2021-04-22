"""
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
"""

from typing import Tuple

from torch import Tensor
import torch

from nndet.core.boxes import batched_nms


def batched_nms_model(
        boxes: Tensor,
        scores: Tensor,
        labels: Tensor,
        weights: Tensor,
        iou_thresh: float,
        *args, **kwargs,
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Model nms for ensembler (same as batched nms with adjusted signature)

    Args:
        boxes: predicted boxes
        scores: predicted scores
        labels: predicted labels
        weights: weight per box
        iou_thresh: IoU threshold for nms
        *args: kept for compatibility
        **kwargs: kept for compatibility

    Returns:
        Tensor: sorted boxes
        Tensor: sorted scores (descending)
        Tensor: sorted labels
        Tensor: sorted weights
    """
    keep = batched_nms(boxes=boxes, scores=scores,
                       idxs=labels, iou_threshold=iou_thresh,
                       )
    return boxes[keep], scores[keep], labels[keep], weights[keep]


def batched_weighted_nms_model(
        boxes: Tensor,
        scores: Tensor,
        labels: Tensor,
        weights: Tensor,
        iou_thresh: float,
        *args, **kwargs,
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Model nms for ensembler (same as batched nms with adjusted signature)

    Args:
        boxes: predicted boxes
        scores: predicted scores
        labels: predicted labels
        weights: weight per box
        iou_thresh: IoU threshold for nms
        *args: kept for compatibility
        **kwargs: kept for compatibility

    Returns:
        Tensor: sorted boxes
        Tensor: sorted scores (descending)
        Tensor: sorted labels
        Tensor: sorted weights
    """
    new_scores = scores * weights
    keep = batched_nms(boxes=boxes, scores=new_scores, idxs=labels, iou_threshold=iou_thresh)
    new_weights = torch.ones_like(weights)
    return boxes[keep], scores[keep], labels[keep], new_weights[keep]
