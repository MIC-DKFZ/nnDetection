# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0


from abc import abstractmethod

import torch
from torch import nn


class ClassCriterion(nn.Module):
    def __init__(self, loss_weight: float) -> None:
        """
        Base class to compute classification cost matrices

        Args:
            loss_weight: weighting for computed loss
        """
        super().__init__()
        self.loss_weight = loss_weight

    @abstractmethod
    def forward(
        self,
        pred_logits: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute class criterion

        Args:
            pred_logits: predicted logits [B * R, C] where B=batch size,
            R=number of predictions, C=number of
                classes
            target_labels: target label for each bounding box [L] where
                L is the number of ground truth objects

        Returns:
            torch.Tensor: cost matrix [B * R, L], where B=batch size,
                R=number of predictions, L is the number of ground truth
                objects
        """
        raise NotImplementedError


class BoxCriterion(nn.Module):
    def __init__(self, loss_weight: float) -> None:
        """
        Base class to compute regression/box cost matrices

        Args:
            loss_weight: weighting for computed loss
        """
        super().__init__()
        self.loss_weight = loss_weight

    @abstractmethod
    def forward(
        self,
        pred_coords: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute box criterion

        Args:
            pred_coords: predicted bounding box coords [B * R, dims * 2]
                where B=batch size, R=number of predictions, dims=number of
                spatial dimensions (format corresponds to model format)
            target_labels: target ground truth boxes [L, dims * 2] where
                L is the number of ground truth objects  (format corresponds
                to model format)

        Returns:
            torch.Tensor: cost matrix [B * R, L], where B=batch size,
                R=number of predictions, L is the number of ground truth
                objects
        """
        raise NotImplementedError
