# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
from loguru import logger
from torch import Tensor
from torch.cuda.amp import autocast
from torchvision.ops.boxes import nms as nms_2d

try:
    from nndet._C import nms as nms_gpu
except ImportError as e:
    logger.warning(
        f"NMS Cuda import failed with {e}, nnDetection was probably not build with GPU support or build failed!"
    )
    nms_gpu = None

import nndet.core.ops_torch as ops_torch


def nms_cpu(boxes, scores, thresh):
    """
    Performs non-maximum suppression for 3d boxes on cpu

    Args:
        boxes: tensor with boxes (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        scores: score for each box [N]
        iou_threshold: threshould when boxes are discarded

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores
    """
    ious = ops_torch.box_iou(boxes, boxes)
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
        boxes: tensor with boxes (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        scores: score for each box [N]
        iou_threshold: threshould when boxes are discarded

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores
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
        Tensor: int64 tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores
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
    masks: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    """
    Model nms for ensembler (same as batched nms with adjusted signature)
    (NMS is always performed on the boxes!)

    Args:
        boxes: predicted boxes
        scores: predicted scores
        labels: predicted labels
        weights: weight per box
        iou_thresh: IoU threshold for nms
        masks: predicted masks

    Returns:
        Tensor: (sorted) postprocessed boxes
        Tensor: (sorted) postprocessed masks. Only returned if masks is not None.
            Skipped otherwise!
        Tensor: (sorted) postprocessed scores (descending)
        Tensor: (sorted) postprocessed labels
        Tensor: (sorted) if weights is not None, corresponding weights, None otherwise
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

    if masks is not None:
        return boxes[keep], masks[keep], scores[keep], labels[keep], _weights
    else:
        return boxes[keep], scores[keep], labels[keep], _weights


def batched_weighted_nms(
    boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    iou_thresh: float,
    weights: Tensor,
    masks: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Uses scores and weights to compute NMS suppression
    Returned scores are the original ones and weights are set to one
    (NMS is always performed on the boxes!)

    Args:
        boxes: predicted boxes
        scores: predicted scores
        labels: predicted labels
        weights: weight per box
        iou_thresh: IoU threshold for nms
        masks: predicted masks

    Returns:
        Tensor: (sorted) postprocessed boxes
        Tensor: (sorted) postprocessed masks. Only returned if masks is not None.
            Skipped otherwise!
        Tensor: (sorted) kept scores.
        Tensor: (sorted) postprocessed labels
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

    if masks is not None:
        return boxes[keep], masks[keep], scores[keep], labels[keep], new_weights[keep]
    else:
        return boxes[keep], scores[keep], labels[keep], new_weights[keep]


def asymmetric_nms(
    boxes: Tensor,
    scores: Tensor,
    iov_threshold: float = 1,
) -> Tensor:
    """
    Performs non-maximum suppression on smaller bounding boxes whose scores are
    lower than the enveloping bounding box

    Args:
        boxes: tensor with boxes (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        scores: score for each box [N]
        iov_threshold: discards all nested bounding boxes with IoV >= iov_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been
            kept by Asymmetric NMS, sorted in decreasing order of scores
    """
    assert 0 <= iov_threshold <= 1
    box_vols = ops_torch.box_area(boxes)
    box_inter = ops_torch.box_inter(boxes, boxes)
    iovs = box_inter / box_vols

    _, _idx = torch.sort(scores, descending=True)

    keep = []
    while _idx.nelement() > 0:
        keep.append(_idx[0])
        # get all elements that were not matched and discard all others.
        non_matches = torch.where((iovs[_idx[0]][_idx] < iov_threshold))[0]
        _idx = _idx[non_matches]
    return torch.tensor(keep).to(boxes).long()


def multiclass_asymmetric_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iov_threshold: float,
) -> Tensor:
    """
    Performs asymmetric non-maximum suppression in a batched fashion.
    Each index value correspond to a category, and Asymmetric NMS
    will not be applied between elements of different categories.

    Args:
        boxes: boxes where Asymmetric NMS will be performed
            (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        scores: scores for each one of the boxes [N]
        idxs: indices of the categories for each one of the boxes. [N]
        iov_threshold: discards all nested bounding boxes with IoV >= iov_threshold

    Returns:
        Tensor: (sorted) postprocessed boxes
        Tensor: (sorted) postprocessed scores (descending)
        Tensor: (sorted) postprocessed labels

    """
    if boxes.numel() == 0:
        return boxes, scores, idxs
    # strategy: in order to perform Asymmetric NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_asym_nms = boxes + offsets[:, None]
    keep = asymmetric_nms(boxes_for_asym_nms, scores, iov_threshold)
    return boxes[keep], scores[keep], idxs[keep]
