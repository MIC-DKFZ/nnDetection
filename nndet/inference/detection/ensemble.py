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

from nndet.core.boxes import batched_nms, nms
from nndet.inference.detection import batched_wbc


def batched_nms_ensemble(
        boxes: Tensor,
        scores: Tensor,
        labels: Tensor,
        weights: Tensor,
        iou_thresh: float,
        *args, **kwargs,
        ) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Ensemble nms for ensembler (same as batched nms with adjusted signature)

    Args:
        boxes: predicted boxes
        scores: predicted scores
        labels: predicted labels
        weights: weight per box (ignored in this function)
        iou_thresh: IoU threshold for nms
        *args: kept for compatibility
        **kwargs: kept for compatibility

    Returns:
        Tensor: boxes
        Tensor: scores
        Tensor: labels
    """
    keep = batched_nms(boxes=boxes, scores=scores,
                       idxs=labels, iou_threshold=iou_thresh,
                       )
    return boxes[keep], scores[keep], labels[keep]


def batched_wbc_ensemble(
        boxes: Tensor,
        scores: Tensor,
        labels: Tensor,
        weights: Tensor,
        iou_thresh: float,
        n_exp_preds: Tensor,
        score_thresh: float,
        *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Ensemble wbc for ensembler (same as batched nms with adjusted signature)

    Args:
        boxes: predicted boxes
        scores: predicted scores
        labels: predicted labels
        weights: weight per box (ignored in this function)
        iou_thresh: IoU threshold for nms
        n_exp_preds: number of expected predictions per box
        score_thresh: minimum score
        *args: kept for compatibility
        **kwargs: kept for compatibility

    Returns:
        Tensor: boxes
        Tensor: scores
        Tensor: labels
    """
    boxes, scores, labels = batched_wbc(
        boxes, scores, labels, weights=weights,
        n_exp_preds=n_exp_preds,
        iou_thresh=iou_thresh,
        score_thresh=score_thresh,
    )
    return boxes, scores, labels


def wbc_nms_no_label_ensemble(
        boxes: Tensor,
        scores: Tensor,
        labels: Tensor,
        weights: Tensor,
        iou_thresh: float,
        n_exp_preds: Tensor,
        score_thresh: float,
        *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Normal wbc -> nms without class labels
    This results in a single prediction per position regardless of the class

    Args:
        boxes: predicted boxes
        scores: predicted scores
        labels: predicted labels
        weights: weight per box (ignored in this function)
        iou_thresh: IoU threshold for nms
        n_exp_preds: number of expected predictions per box
        score_thresh: minimum score
        *args: kept for compatibility
        **kwargs: kept for compatibility

    Returns:
        Tensor: boxes
        Tensor: scores
        Tensor: labels
    """
    boxes, scores, labels = batched_wbc(
        boxes, scores, labels, weights=weights,
        n_exp_preds=n_exp_preds,
        iou_thresh=iou_thresh,
        score_thresh=score_thresh,
    )
    keep = nms(boxes, scores, iou_thresh)
    return boxes[keep], scores[keep], labels[keep]
