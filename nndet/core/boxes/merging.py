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

from abc import ABC, abstractmethod
from typing import Tuple, Callable, List

import torch
from torch import Tensor

from nndet.detection.boxes.utils import box_iou


def weighted_merging(boxes: torch.Tensor, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Weighted mean merging of boxes

    Args:
        boxes: boxes [N, 2*dim](x0, x1, y0, y1, z0, z1)
        scores: weight for each box [N]

    Returns:
        torch.Tensor: new box [2 * dim]
        torch.Tensor new score [1]
    """
    new_boxes = (boxes * scores[:, None]).sum(dim=0) / scores.sum()
    new_scores = scores.mean()
    return new_boxes, new_scores


class Merger(ABC):
    def __init__(self,
                 iou_th: float,
                 iou_fn: Callable[[Tensor, Tensor], Tensor] = box_iou,
                 ):
        """
        Generic Merger Interface

        Args:
            iou_th: min IoU between instances to merge
            iou_fn: similarity function which computes overlap between
                instances
        """
        super().__init__()
        self.iou_th = iou_th
        self.iou_fn = iou_fn

    @abstractmethod
    def merge(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Merge multiple boxes

        Returns:
            Tensor: new set of instances
            Tensor: new set of scores
            Tensor: new set of labels
        """
        raise NotImplementedError


class GreedyIoUBoxMerger(Merger):
    def __init__(self,
                 boxes: Tensor,
                 slices: Tensor,
                 scores: Tensor,
                 labels: Tensor,
                 iou_th: float,
                 iou_fn: Callable[[Tensor, Tensor], Tensor] = box_iou,
                 neighbor_slices: int = 1,
                 ):
        """
        Merge 2D Boxes from slices to 3D boxes with greedy IoU tracking

        Args:
            boxes: boxes to merge
            slices: slice for each box
            scores: score for each instance
            labels: label for each instance
            iou_th: min IoU between instances to merge
            iou_fn: similarity function which computes overlap between
                instances
            neighbor_slices: number of neighboring slices to inspect
        """
        super().__init__(
            iou_th=iou_th,
            iou_fn=iou_fn,
        )
        if neighbor_slices < 1:
            raise ValueError(f"neighbor_slices must be at least one, found {neighbor_slices}")
        if iou_th < 0 or iou_th > 1:
            raise ValueError(f"IoU threshold needs to be within [0,1], found {iou_th}")
        if not isinstance(boxes, torch.Tensor):
            raise ValueError(f"Wrong type for boxes, got {type(boxes)} expected Tensor.")
        if not isinstance(scores, torch.Tensor):
            raise ValueError(f"Wrong type for scores, got {type(boxes)} expected Tensor.")
        if not isinstance(labels, torch.Tensor):
            raise ValueError(f"Wrong type for labels, got {type(boxes)} expected Tensor.")
        if boxes.shape[0] != len(scores) or boxes.shape[0] != len(labels):
            raise ValueError("Every Box needs a label and a score")
        self.boxes = boxes
        self.slices = slices
        self.scores = scores
        self.labels = labels
        self.neighbor_slices = neighbor_slices

    def merge(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Merge 2d boxes to 3d boxes

        Returns:
            Tensor: new set of boxes
            Tensor: new set of scores
            Tensor: new set of labels
        """
        if self.boxes.numel() == 0:
            return self.boxes.view(-1, 6), self.scores, self.labels

        _, idx_sorted = self.scores.sort(descending=True)
        idx_sorted = idx_sorted.detach().numpy().tolist()
        boxes_3d, scores_3d, labels_3d = [], [], []
        while idx_sorted:  # iterate while there are unmatched boxes
            seed_index = idx_sorted[0]  # get highest scoring box
            current_boxes, current_slices, current_scores, current_labels = [], [], [], []

            idx_selected = self.select_idx_subset(
                idx=idx_sorted,
                seed_index=seed_index,
            )
            tracked_indices, tracked_boxes, tracked_slices, tracked_scores, tracked_labels = \
                self.build_track(
                    seed_index=seed_index,
                    idx_list=idx_sorted,
                    idx_selected=idx_selected,
                )
            current_boxes.extend(tracked_boxes)
            current_slices.extend(tracked_slices)
            current_scores.extend(tracked_scores)
            current_labels.extend(tracked_labels)
            idx_sorted = [i for i in idx_sorted if i not in tracked_indices]

            box_tracked, score_tracked, label_tracked = self.merge_track(
                boxes=current_boxes,
                scores=current_scores,
                labels=current_labels,
                slices=current_slices
            )

            boxes_3d.append(box_tracked)
            scores_3d.append(score_tracked)
            labels_3d.append(label_tracked)
        return torch.stack(boxes_3d, dim=0), torch.stack(scores_3d), torch.stack(labels_3d)

    def select_idx_subset(self,
                          idx: List[int],
                          seed_index: int,
                          ) -> List[int]:
        """
        Selects all boxes which match the label of the seed box

        Args:
            idx: all active indices
            seed_index: seed index

        Returns:
            List[int]: indices of boxes which match the label of the seed box
        """
        idx_correct_label = [i for i in idx if self.labels[i] == self.labels[seed_index]]
        return idx_correct_label

    def build_track(self,
                    seed_index: int,
                    idx_list: List[int],
                    idx_selected: List[int],
                    ) -> Tuple[List[int], List[Tensor], List[int], List[Tensor], List[Tensor]]:
        """
        Select boxes with sufficient overlap in a greedy way

        Args:
            seed_index: index of seed box
            idx_list: list with all active indices
            idx_selected: list with indices to use for tracking

        Returns:
            List[int]: indices of tracked boxes
            List[Tensor]: selected boxes [R, 4]
            List[int]: selected slices [R]
            List[Tensor]: selected scores [R]
            List[Tensor]: selected labels [R] should all be the same.
        """
        tracked_indices = [seed_index]
        tracked_boxes = [self.boxes[seed_index]]
        tracked_slices = [self.slices[seed_index]]
        tracked_scores = [self.scores[seed_index]]
        tracked_labels = [self.labels[seed_index]]

        for direction in [1, -1]:
            matched = True
            box_index = seed_index

            while matched:
                matched = False
                for nb in range(1, self.neighbor_slices + 1):
                    expansion_index = [int(i) for i in idx_selected if
                                       self.slices[i] == self.slices[box_index] + nb * direction]

                    if not expansion_index:
                        continue  # continue to next slice

                    match_quality_matrix = self.iou_fn(
                        self.boxes[[int(box_index)]], self.boxes[expansion_index])  # 1 x M

                    max_iou, max_iou_index = match_quality_matrix.max(dim=1)
                    if max_iou > self.iou_th:
                        # add to cluster
                        idx_list_index = idx_list.index(expansion_index[max_iou_index])
                        box_index = idx_list[idx_list_index]

                        tracked_indices.append(box_index)
                        tracked_boxes.append(self.boxes[box_index])
                        tracked_slices.append(self.slices[box_index])
                        tracked_scores.append(self.scores[box_index])
                        tracked_labels.append(self.labels[box_index])
                        matched = True
                        break  # found matching slice; break inner slice loop
        return tracked_indices, tracked_boxes, tracked_slices, tracked_scores, tracked_labels

    @staticmethod
    def merge_track(
            boxes: List[Tensor],
            scores: List[Tensor],
            labels: List[Tensor],
            slices: List[int],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Merge selected boxes, scores and labels to a new instance
        Boxes are merged with their max extend.
        Median scores is used and the first label is used (all labels should
        be the same)

        Args:
            boxes: selected boxes [N, 4](x0, x1, y0, y1)
            slices: slice of each box [N]
            scores: selected scores [R]
            labels: selected labels [S]

        Returns:
            Tensor: merged 3D box [6][x0, x1, y0, y1, z0,z0)
            Tensor: merged score [1]
            Tensor: merges label [1]

        Notes:
            Boxes, labels and scores are merged independently so they they can
            have different number of elements
        """
        _boxes = torch.stack(boxes, dim=0)
        _scores = torch.stack(scores, dim=0)
        _labels = labels[0]  # all labels are the same

        box_3d = torch.tensor([
                min(slices),
                min(_boxes[:, 0]),
                max(slices) + 1,
                max(_boxes[:, 2]),
                min(_boxes[:, 1]),
                max(_boxes[:, 3]),
            ])
        score_3d = _scores.median()
        label_3d = _labels
        return box_3d, score_3d, label_3d


class VoteLabelGreedyIoUBoxMerger(GreedyIoUBoxMerger):
    def select_idx_subset(self,
                          idx: List[int],
                          seed_index: int,
                          ) -> List[int]:
        """
        Ignores the label and returns all boxes

        Args:
            idx: all active indices
            seed_index: seed index

        Returns:
            List[int]: indices of boxes which match the label of the seed box
        """
        return idx

    def build_track(self,
                    seed_index: int,
                    idx_list: List[int],
                    idx_selected: List[int],
                    ) -> Tuple[List[int], List[Tensor], List[int], List[Tensor], List[Tensor]]:
        """
        Select boxes with sufficient overlap in a greedy way.
        In this case we assume that one box can match to multiple other
        boxes in the same and neighboring planes (due to the class
        independence). The boxes of will be merged by
        an weighted average inside the respected slice.
        The scores and labels will  be appended

        Args:
            seed_index: index of seed box
            idx_list: list with all active indices
            idx_selected: list with indices to use for tracking

        Returns:
            List[int]: indices of tracked boxes
            List[Tensor]: selected boxes [R, 4]
            List[int]: selected slices [R]
            List[Tensor]: selected scores [X]; X != R in case of multiple
                matches
            List[Tensor]: selected labels [X] should all be the same.; X != R
                in case of multiple matches
        """
        tracked_indices, tracked_boxes, tracked_slices, tracked_scores, tracked_labels = [], [], [], [], []
        for direction in [0, 1, -1]:
            matched = True
            box_index = seed_index

            while matched:
                matched = False
                for nb in range(1, self.neighbor_slices + 1):
                    expansion_index = [int(i) for i in idx_selected if
                                       self.slices[i] == self.slices[box_index] + nb * direction]

                    if not expansion_index:
                        continue  # continue to next slice

                    match_quality_matrix = self.iou_fn(
                        self.boxes[[int(box_index)]], self.boxes[expansion_index])  # 1 x M
                    matched_idx = torch.nonzero(match_quality_matrix > self.iou_th, as_tuple=True)[1]

                    # we need to keep track of the boxes in this slice to merge them
                    matched_boxes, matched_scores = [], []
                    for midx in matched_idx:
                        # remove from active boxes
                        idx_list_index = idx_list.index(expansion_index[midx])
                        box_index = idx_list[idx_list_index]

                        tracked_indices.append(box_index)
                        tracked_slices.append(self.slices[box_index])
                        tracked_scores.append(self.scores[box_index])
                        tracked_labels.append(self.labels[box_index])

                        matched_boxes.append(self.boxes[box_index])
                        matched_scores.append(self.scores[box_index])

                    if matched_boxes:
                        if len(matched_boxes) > 1:
                            merged_box, _ = weighted_merging(
                                torch.stack(matched_boxes, dim=0), torch.stack(matched_scores))
                        else:
                            merged_box = matched_boxes[0]
                        tracked_boxes.append(merged_box)
                        matched = True
                        break  # found matching slice; break inner slice loop
                if direction == 0:
                    break  # break loop after we check the current slice once
        return tracked_indices, tracked_boxes, tracked_slices, tracked_scores, tracked_labels

    @staticmethod
    def merge_track(
            boxes: List[Tensor],
            scores: List[Tensor],
            labels: List[Tensor],
            slices: List[int],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Merge selected boxes, scores and labels to a new instance
        Boxes are merged with their max extend.
        The label is determined by a weighted majority voting. The
        score is the median score of the selected label.

        Args:
            boxes: selected boxes [N, 4](x0, x1, y0, y1)
            slices: slice of each box [N]
            scores: selected scores [R]
            labels: selected labels [S]

        Returns:
            Tensor: merged 3D box [6][x0, x1, y0, y1, z0,z0)
            Tensor: merged score [1]
            Tensor: merges label [1]

        Notes:
            Boxes, labels and scores are merged independently so they they can
            have different number of elements
        """
        _boxes = torch.stack(boxes, dim=0)
        _scores = torch.stack(scores, dim=0)
        _labels = torch.stack(labels, dim=0)

        box_3d = torch.tensor([
                min(slices),
                min(_boxes[:, 0]),
                max(slices) + 1,
                max(_boxes[:, 2]),
                min(_boxes[:, 1]),
                max(_boxes[:, 3]),
            ])

        label_counts = _labels.int().bincount(weights=_scores)
        label_3d = torch.argmax(label_counts).float()  # bins indicate the correct label

        score_3d = _scores[_labels == label_3d].median()
        return box_3d, score_3d, label_3d
