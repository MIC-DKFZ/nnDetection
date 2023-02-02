# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# SimpleClassCriterionSoftmax was adapted from original DETR code
# SPDX-FileCopyrightText: 2020 Facebook
# SPDX-License-Identifier: Apache-2.0
#
# FocalClassCriterionSigmoid was adapted from Deformable DETR code
# SPDX-FileCopyrightText: 2020 SenseTime
# SPDX-License-Identifier: Apache-2.0


import torch

from nndet.core.boxes.criterions.base import ClassCriterion


class SimpleClassCriterionSoftmax(ClassCriterion):
    def __init__(self, loss_weight: float) -> None:
        """
        Comute simple class criterion with softmax logits

        Args:
            loss_weight: weighting for computed loss
        """
        super().__init__(loss_weight=loss_weight)
        self.logits_convert_fn = torch.nn.Softmax(dim=-1)

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute softmax based class criterion

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
        pred_probs = self.logits_convert_fn(pred_logits)
        return self.loss_weight * -1 * pred_probs[:, target_labels]


class SimpleClassCriterionSigmoid(ClassCriterion):
    def __init__(self, loss_weight: float) -> None:
        """
        Comute simple class criterion with softmax logits

        Args:
            loss_weight: weighting for computed loss
        """
        super().__init__(loss_weight=loss_weight)
        self.logits_convert_fn = torch.nn.Sigmoid()

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute softmax based class criterion

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
        pred_probs = self.logits_convert_fn(pred_logits)
        target_labels_idx = target_labels - 1

        return self.loss_weight * -1 * pred_probs[:, target_labels_idx]


class FocalClassCriterionSigmoid(ClassCriterion):
    def __init__(
        self,
        alpha: float,
        gamma: float,
        loss_weight: float,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(loss_weight)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.logits_convert_fn = torch.nn.Sigmoid()

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute softmax based class criterion

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
        pred_probs = self.logits_convert_fn(pred_logits)
        target_labels_idx = target_labels - 1

        neg_cost_class = (1 - self.alpha) * (pred_probs**self.gamma) * (-(1 - pred_probs + self.eps).log())
        pos_cost_class = self.alpha * ((1 - pred_probs) ** self.gamma) * (-(pred_probs + self.eps).log())

        cost_class = pos_cost_class[:, target_labels_idx] - neg_cost_class[:, target_labels_idx]
        return cost_class
