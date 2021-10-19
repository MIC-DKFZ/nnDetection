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

from typing import Optional, Tuple

import torch
from loguru import logger
from torch import Tensor
from torch.cuda.amp import autocast
from torchvision.ops.boxes import nms as nms_2d

try:
    from nndet._C import nms as nms_gpu
except ImportError:
    logger.warning("nnDetection was not build with GPU support!")
    nms_gpu = None
from nndet.core.boxes.ops import box_iou


def nms_cpu(boxes, scores, thresh):
    """
    Performs non-maximum suppression for 3d boxes on cpu

    Args:
        boxes (Tensor): tensor with boxes (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        scores (Tensor): score for each box [N]
        iou_threshold (float): threshould when boxes are discarded

    Returns:
        keep (Tensor): int64 tensor with the indices of the elements that have been kept by NMS,
            sorted in decreasing order of scores
    """
    ious = box_iou(boxes, boxes)
    _, _idx = torch.sort(scores, descending=True)

    keep = []
    while _idx.nelement() > 0:
        keep.append(_idx[0])
        # get all elements that were not matched and discard all others.
        non_matches = torch.where((ious[_idx[0]][_idx] <= thresh))[0]
        _idx = _idx[non_matches]
    return torch.tensor(keep).to(boxes).long()


@autocast(enabled=False)
def nms(
    boxes: Tensor,
    scores: Tensor,
    iou_threshold: float,
) -> Tensor:
    """
    Performs non-maximum suppression

    Args:
        boxes (Tensor): tensor with boxes (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        scores (Tensor): score for each box [N]
        iou_threshold (float): threshould when boxes are discarded

    Returns:
        keep (Tensor): int64 tensor with the indices of the elements that have been kept by NMS,
            sorted in decreasing order of scores
    """
    if boxes.shape[1] == 4:
        # prefer torchvision in 2d because they have c++ cpu version
        nms_fn = nms_2d
        # nms_fn = nms_cpu
    else:
        if boxes.is_cuda:
            nms_fn = nms_gpu
        else:
            nms_fn = nms_cpu
    return nms_fn(boxes.float(), scores.float(), iou_threshold)


def _batched_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    """
    Performs non-maximum suppression in a batched fashion.
    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes: boxes where NMS will be performed
            (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        scores: scores for each one of the boxes [N]
        idxs: indices of the categories for each one of the boxes. [N]
        iou_threshold:  discards all overlapping boxes with IoU > iou_threshold

    Returns
        keep: int64 tensor with the indices of the elements that have been kept by NMS,
            sorted in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    return nms(boxes_for_nms, scores, iou_threshold)


def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    iou_thresh: float,
    weights: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    """
    Model nms for ensembler (same as batched nms with adjusted signature)

    Args:
        boxes: predicted boxes
        scores: predicted scores
        labels: predicted labels
        weights: weight per box
        iou_thresh: IoU threshold for nms

    Returns:
        Tensor: postprocessed boxes
        Tensor: postprocessed scores (descending)
        Tensor: postprocessed labels
        Tensor: if weights is not None, corresponding weights, None otherwise
    """
    keep = _batched_nms(
        boxes=boxes,
        scores=scores,
        idxs=labels,
        iou_threshold=iou_thresh,
    )

    if weights is not None:
        _weights = weights[keep]
    else:
        _weights = None

    return boxes[keep], scores[keep], labels[keep], _weights


def batched_weighted_nms(
    boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    iou_thresh: float,
    weights: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Uses scores and weights to compute NMS suppression
    Returned scores are the original ones

    Args:
        boxes: predicted boxes
        scores: predicted scores
        labels: predicted labels
        weights: weight per box
        iou_thresh: IoU threshold for nms

    Returns:
        Tensor: postprocessed boxes
        Tensor: kept scores.
        Tensor: postprocessed labels
        Tensor: vector filled with ones.
    """
    _scores = scores * weights
    keep = _batched_nms(
        boxes=boxes,
        scores=_scores,
        idxs=labels,
        iou_threshold=iou_thresh,
    )
    new_weights = torch.ones_like(weights)

    return boxes[keep], scores[keep], labels[keep], new_weights[keep]
