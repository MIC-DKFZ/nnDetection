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

import torch

from torch import Tensor
from typing import Tuple

from torch._C import device

from nndet.core.boxes import box_iou, box_area


__all__ = ["batched_wbc", "wbc"]


def batched_wbc(
    boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    weights: Tensor,
    iou_thresh: float,
    n_exp_preds: Tensor,
    score_thresh: float,
    use_area: bool = False,
    missing_weight: float = 1.,
    ) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Computed weighted box clustering per class

    Args:
        boxes: predicted boxes (x1, y1, x2, y2, (z1, z2)) [N, dims * 2]
        scores: predicted scores [N]
        labels: predicted labels [N]
        weights: weight for each box [N] (gaussian weighting of boxes near
            corners need to be included in this weight)
        iou_thresh: iou threshold used for clustering boxes
        n_exp_preds: number of expected predictions per box (computed as the
            mean number predictions inside the bounding box)
        score_thresh: minimum score of predictions after clustering
        use_area: assigns higher weights to larger boxes based on
            empirical observations indicating an increase in image
            evidence from larger areas.
        missing_weight: weight for score dampening when predictions are missing

    Returns:
        Tensor: clustered boxes
        Tensor: clustered scores
        Tensor: labels
    """
    clustered_boxes = []
    clustered_scores = []
    clustered_labels = []
    for label in labels.unique():
        _labels_mask = labels == label
        _boxes = boxes[_labels_mask]
        _scores = scores[_labels_mask]
        _weights = weights[_labels_mask]
        _n_exp_preds = n_exp_preds[_labels_mask]

        b, s = wbc(_boxes, _scores,
                   weights=_weights, n_exp_preds=_n_exp_preds,
                   iou_thresh=iou_thresh, score_thresh=score_thresh,
                   use_area=use_area,
                   missing_weight=missing_weight,
                   )

        clustered_boxes.append(b)
        clustered_scores.append(s)
        clustered_labels.append(torch.empty_like(s).fill_(label))
    if clustered_boxes:
        return (torch.cat(clustered_boxes, dim=0),
                torch.cat(clustered_scores, dim=0),
                torch.cat(clustered_labels, dim=0))
    else:
        return (torch.tensor([]).view(-1, boxes.shape[1]),
                torch.tensor([]).view(-1),
                torch.tensor([]).view(-1))


def wbc(
    boxes: Tensor,
    scores: Tensor,
    weights: Tensor,
    n_exp_preds: Tensor,
    iou_thresh: float,
    score_thresh: float,
    use_area: bool = True,
    missing_weight: float = 1.,
    ) -> Tuple[Tensor, Tensor]:
    """
    Weighted box clustering

    Args:
        boxes: tensor with boxes (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        scores: score for each box [N]
        weights: additional weights for boxes [N]
        n_exp_preds: expected number of predictions per box
        iou_thresh: iou threshold for determining clusters of boxes which are
            combined
        score_thresh: minimum scores of boxes after consolidation
        use_area: assigns higher weights to larger boxes based on
            empirical observations indicating an increase in image
            evidence from larger areas.
        missing_weight: weight for score dampening when predictions are missing

    Returns:
        Tensor: consolidated boxes
        Tensor: consolidated scores
    """
    ious = box_iou(boxes, boxes)

    if use_area:
        areas = box_area(boxes)
        weights = weights * areas

    _, idx_pool = torch.sort(scores, descending=True)

    new_boxes, new_scores = [], []
    while idx_pool.nelement() > 0:
        # build cluster
        highest_scoring_id = idx_pool[0]
        matches = torch.where(ious[highest_scoring_id][idx_pool] > iou_thresh)[0].flatten()
        box_idx = idx_pool[matches]

        # compute new scores
        n_expected = n_exp_preds[box_idx].float().mean()
        new_box, new_score = compute_cluster_consolidation(
            boxes[box_idx], scores[box_idx],
            weights=weights[box_idx],
            ious=ious[highest_scoring_id][box_idx],
            n_expected=n_expected,
            n_found=len(box_idx),
            missing_weight=missing_weight,
            )

        if new_score > score_thresh:
            new_boxes.append(new_box)
            new_scores.append(new_score)

        # get all elements that were not matched and discard all others.
        non_matches = torch.where(ious[highest_scoring_id][idx_pool] <= iou_thresh)[0].flatten()
        idx_pool = idx_pool[non_matches]
    if new_boxes:
        return torch.stack(new_boxes, dim=0), torch.cat(new_scores, dim=0)
    else:
        return torch.tensor([]).view(-1, boxes.shape[1]).to(boxes), torch.tensor([]).view(-1).to(scores)


def compute_cluster_consolidation(
    boxes: Tensor,
    scores: Tensor,
    weights: Tensor,
    ious: Tensor,
    n_expected: Tensor,
    n_found: int,
    missing_weight: float,
    ) -> Tuple[Tensor, Tensor]:
    """
    Consolidate predictions of a single cluster

    Args:
        boxes: boxes of a single cluster (x1, y1, x2, y2, (z1, z2) [N, dims * 2]
        scores: scores of a single cluster [N]
        weights: weights for boxes of a single cluster [N]
        ious: ious with recard to highest scoring box in a single cluster [N]
        n_expected: expected number of predictions
        n_found: number of predictions
        missing_weight: weight for score dampening when predictions are missing

    Returns:
        Tensor: new boxes (x1, y1, x2, y2, (z1, z2) [N, dims * 2]
        Tensor: new scores [N]
    """
    # compute new score
    match_score_weights = ious * weights
    match_scores = match_score_weights * scores

    n_missing_preds = torch.max(torch.tensor([0.], device=n_expected.device),
                                (n_expected - n_found).float())
    denom = match_score_weights.sum() + n_missing_preds * match_score_weights.mean() * missing_weight
    consolidated_score = match_scores.sum() / denom

    consolidated_boxes = (boxes * match_scores.reshape(-1, 1)).sum(dim=0) / match_scores.sum()
    return consolidated_boxes, consolidated_score


def compute_cluster_consolidation2(
    boxes: Tensor,
    scores: Tensor,
    weights: Tensor,
    ious: Tensor,
    n_expected: Tensor,
    n_found: int,
    missing_weight: float,
    ) -> Tuple[Tensor, Tensor]:
    """
    Consolidate predictions of a single cluster

    Args:
        boxes: boxes of a single cluster (x1, y1, x2, y2, (z1, z2) [N, dims * 2]
        scores: scores of a single cluster [N]
        weights: weights for boxes of a single cluster [N]
        ious: ious with recard to highest scoring box in a single cluster [N]
        n_expected: expected number of predictions
        n_found: number of predictions
        missing_weight: weight for score dampening when predictions are missing

    Returns:
        Tensor: new boxes (x1, y1, x2, y2, (z1, z2) [N, dims * 2]
        Tensor: new scores [N]
    """
    # select num expected predictions from ious & score weihted score
    topk_score = ious * weights * scores
    topk_weighted_scores, topk_idx = topk_score.topk(min(len(scores), int(n_expected)))

    boxes = boxes[topk_idx]
    scores = scores[topk_idx]
    n_missing_preds = torch.max(torch.tensor([0.], device=n_expected.device),
                            (n_expected - n_found).float())
    
    # weigh predictions with high ious higher, penalty term for missing predictions
    consolidated_score = scores.mean() * (1 - missing_weight * n_missing_preds / n_expected)
    consolidated_boxes = (boxes * topk_weighted_scores.reshape(-1, 1)).sum(dim=0) / topk_weighted_scores.sum()

    return consolidated_boxes, consolidated_score
