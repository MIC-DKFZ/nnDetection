# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import torch

from nndet.losses.classification.functional.poly1 import (
    poly1_bce_with_logits_jit as poly1_bce_with_logits,
)
from nndet.losses.classification.functional.poly1 import (
    poly1_focal_loss_with_logits_jit as poly1_focal_loss_with_logits,
)
from nndet.losses.ops import SigmoidBaseLoss


class Poly1FocalLossWithLogits(SigmoidBaseLoss):
    def __init__(
        self,
        gamma: float = 2,
        alpha: float = -1,
        epsilon: float = -1,
        loss_fp32: bool = False,
        loss_weight: float = 1.0,
        reduction: str = "sum",
        smoothing: float = 0.0,
    ):
        """
        Poly1 Focal loss with multiple classes
        (internally uses one hot encoding and sigmoid)

        Poly1 Focal-Loss
        https://openreview.net/forum?id=gSdSJoenupI

        Args:
            gamma: balance easy and hard examples in focal loss
            alpha: balance positive and negative samples [0, 1] (increasing
                alpha increase weight of foreground classes (better recall))
            epsilon: epsilon of poly term.
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            reduction: 'mean'|'sum'|'none'
                mean: mean of loss over entire batch
                sum: sum of loss over entire batch
                none: no reduction
            smoothing: optionally apply label smoothing. Default 0.0 -> no label
                smoothing.
        """
        super().__init__(
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
            reduction=reduction,
            smoothing=smoothing,
        )
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def comp_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss with subclass loss function

        Args:
            logits: predicted logits [N, C, dims], where N is the batch size,
                C number of classes, dims are arbitrary spatial dimensions
                (background classes should be located at channel 0 if
                ignore background is enabled)
            targets: ont-hot targets [N, C, dims], where N is the batch size,
                C number of classes, dims are arbitrary spatial dimensions

        Returns:
            torch.Tensor: loss
        """
        return poly1_focal_loss_with_logits(
            logits,
            targets,
            gamma=self.gamma,
            alpha=self.alpha,
            epsilon=self.epsilon,
            reduction=self.reduction,
        )


class Poly1BCEWithLogits(SigmoidBaseLoss):
    def __init__(
        self,
        alpha: float = -1,
        epsilon: float = -1,
        loss_fp32: bool = False,
        loss_weight: float = 1.0,
        reduction: str = "sum",
        smoothing: float = 0.0,
    ):
        """
        Poly1 BCE loss with multiple classes
        (internally uses one hot encoding and sigmoid)

        Poly1 BCE
        https://openreview.net/forum?id=gSdSJoenupI

        Args:
            gamma: balance easy and hard examples in focal loss
            alpha: balance positive and negative samples [0, 1] (increasing
                alpha increase weight of foreground classes (better recall))
            epsilon: epsilon of poly term.
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            reduction: 'mean'|'sum'|'none'
                mean: mean of loss over entire batch
                sum: sum of loss over entire batch
                none: no reduction
            smoothing: optionally apply label smoothing. Default 0.0 -> no label
                smoothing.
        """
        super().__init__(
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
            reduction=reduction,
            smoothing=smoothing,
        )
        self.alpha = alpha
        self.epsilon = epsilon

    def comp_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss with subclass loss function

        Args:
            logits: predicted logits [N, C, dims], where N is the batch size,
                C number of classes, dims are arbitrary spatial dimensions
                (background classes should be located at channel 0 if
                ignore background is enabled)
            targets: ont-hot targets [N, C, dims], where N is the batch size,
                C number of classes, dims are arbitrary spatial dimensions

        Returns:
            torch.Tensor: loss
        """
        return poly1_bce_with_logits(
            logits,
            targets,
            alpha=self.alpha,
            reduction=self.reduction,
            epsilon=self.epsilon,
        )
