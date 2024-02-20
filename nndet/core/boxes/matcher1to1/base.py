# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
import os
from abc import abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from nndet.core.boxes.criterions.base import BoxCriterion, ClassCriterion


class BaseMatcher(nn.Module):
    # EMPTY_IMG = -1

    def __init__(
        self,
        class_criterion: Sequence[ClassCriterion],
        box_criterion: Sequence[BoxCriterion],
    ) -> None:
        """
        BaseClass for matcher which perform matching based on indices (like
        in DETR, usually bipartit). The output represent the matched indices
        between the predictions and ground truth objects. Usually
        there should be more predictions than ground truth objects and thus
        several predictions will be unmatched (i.e. not part of the
        returned indices). For images without ground truth objects
        the function will return tuples with `None`.

        Args:
            class_criterion: the class criterion is used to compute the weight
                matrix for the classification predictions
            box_criterion: the box criterion is used to compute the weight
                matrix for the regression. The format of the bounding boxes
                need to be defined from the outside module and the criterion.
                The matcher does not take care of the format.
        """
        super().__init__()
        self.class_criterion = class_criterion
        self.box_criterion = box_criterion
        self.extended_logging = os.getenv("det_extended_logging", 0)

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_coords: torch.Tensor,
        target_boxes: List[torch.Tensor],
        target_labels: List[torch.Tensor],
    ) -> Tuple[List[Tuple[Optional[Tensor], Optional[Tensor]]], Dict[str, torch.Tensor]]:
        """
        Perform matching over whole batch

        Args:
            pred_logits: predicted class logits from model [B, R, C]
                where B=batch size, R=number of predictions, C=number of
                classes
            pred_coords: predicted bounding boxes coordinates from model
                [B, R, dims * 2] where B=batch size, R=number of predictions
                dims=number of spatial dimensions
            target_boxes: target ground truth boxes (not encoded)
                List([L, dims * 2]) where L is the number of ground truth boxes
                in each image, dims is the number of spatial dimensions and
                the length of the list corresponds to the batch size
            target_labels: target labels for each box List([L]) where L is the
                number of ground truth boxes in each image and
                the length of the list corresponds to the batch size

        Returns:
            List[Tuple[Tensor, Tensor]]: returns the matched indices for a
                batch. The first tensor contains the selected predictions
                (in order) and the second tensor contains the selected ground
                truth objects (in order). It holds for each elements:
                len(index_i) = len(index_j) = min(num_pred, num_target_boxes)
                Entries with None correspond to images without ground truth
                objects.
            Optional[Dict]: Dict containing the matching cost
        """
        # Filter out patches with no boxes in them
        num_boxes = 0
        mask = []
        masked_boxes = []
        masked_labels = []
        for batch_idx in range(len(target_labels)):
            if target_labels[batch_idx].numel() == 0:
                mask.append(False)
            else:
                num_boxes += len(target_labels[batch_idx])
                masked_boxes.append(target_boxes[batch_idx])
                masked_labels.append(target_labels[batch_idx])
                mask.append(True)

        if num_boxes > 0:
            masked_indices, log_dict = self.match(
                pred_logits=pred_logits[mask],
                pred_coords=pred_coords[mask],
                target_boxes=masked_boxes,
                target_labels=masked_labels,
            )
        else:
            masked_indices = []
            log_dict = {}

        indices = self.unmask_indices(mask, masked_indices)
        return indices, log_dict

    @classmethod
    def unmask_indices(
        cls, mask: List[bool], masked_indices: List[Tensor]
    ) -> List[Tuple[Optional[Tensor], Optional[Tensor]]]:
        """
        Insert entries which were previously masked out. Custom entries
        with `cls.EMPTY_IMG` will be used to uniquely identify the
        added entries.

        Args:
            mask: mask with entries indicating which entries were filtered.
                `True` corresponds to batch elements with at least one ground
                truth box.
            masked_indices: indices which were produced on the masked
                batch elements.

        Returns:
            List[Tuple[Tensor, Tensor]]: returns the matched indices for a
                batch. The first tensor contains the selected predictions
                (in order) and the second tensor contains the selected ground
                truth objects (in order). It holds for each elements:
                len(index_i) = len(index_j) = min(num_pred, num_target_boxes)
                Entries with None correspond to images without ground truth
                objects.
        """
        indices = []
        add_missing = 0
        for i, b in enumerate(mask):
            if b:
                # If the patch is non-empty, append the index from the matcher
                indices.append(masked_indices[i - add_missing])
            else:
                # if the patch was empty, it was not used during the matching, so we have to add +1 to add_missing
                add_missing += 1
                indices.append((None, None))
        return indices

    @abstractmethod
    @torch.no_grad()
    def match(
        self,
        pred_logits: torch.Tensor,
        pred_coords: torch.Tensor,
        target_boxes: List[torch.Tensor],
        target_labels: List[torch.Tensor],
    ) -> List[Tuple[Tensor, Tensor]]:
        """
        Perform matching over batch elements with at least one ground truth
        element in them

        Args:
            pred_logits: predicted class logits from model [B, R, C]
                where B=batch size, R=number of predictions, C=number of
                classes
            pred_coords: predicted bounding boxes coordinates from model
                [B, R, dims * 2] where B=batch size, R=number of predictions
                dims=number of spatial dimensions
            target_boxes: target ground truth boxes (not encoded)
                List([L, dims * 2]) where L is the number of ground truth boxes
                in each image, dims is the number of spatial dimensions and
                the length of the list corresponds to the batch size
            target_labels: target labels for each box List([L]) where L is the
                number of ground truth boxes in each image and
                the length of the list corresponds to the batch size

        Returns:
            List[Tuple[Tensor, Tensor]]: returns the matched indices for a
                batch. The first tensor contains the selected predictions
                (in order) and the second tensor contains the selected ground
                truth objects (in order). It holds for each elements:
                len(index_i) = len(index_j) = min(num_pred, num_target_boxes)
        """
        raise NotImplementedError()
