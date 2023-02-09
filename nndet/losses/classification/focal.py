# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import torch

from nndet.losses.classification.functional.asymfocal import (
    asymmetric_focal_loss_with_logits,
)
from nndet.losses.classification.functional.focal import focal_loss_with_logits
from nndet.losses.ops import SigmoidBaseLoss


class FocalLossWithLogits(SigmoidBaseLoss):
    def __init__(
        self,
        gamma: float = 2,
        alpha: float = -1,
        loss_fp32: bool = False,
        loss_weight: float = 1.0,
        reduction: str = "sum",
        smoothing: float = 0.0,
    ):
        """
        Focal loss with multiple classes (uses one hot encoding and sigmoid)

        Args:
            gamma: balance easy and hard examples in focal loss
            alpha: balance positive and negative samples [0, 1] (increasing
                alpha increase weight of foreground classes (better recall))
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

    def comp_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss with subclass loss function

        Args:
            logits: logits for all foreground classes [*, C]
                * are arbitrary spatial dimensions, C is the number of
                foreground classes
            target: target classes. 0 is treated as background, >0 are
                treated as foreground classes. [*] where * are arbitrary
                spatial dimensions

        Returns:
            torch.Tensor: loss
        """
        return focal_loss_with_logits(
            logits,
            targets,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=self.reduction,
        )


class AsymmetricFocalLossWithLogits(SigmoidBaseLoss):
    def __init__(
        self,
        gamma: float = 2,
        alpha: float = 1,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        reduction: str = "mean",
        smoothing: float = 0.0,
    ):
        """
        Asymmetric Focal loss
        https://arxiv.org/abs/2008.13367
        and https://arxiv.org/abs/1907.10982

        Args:
            gamma: balance easy and hard examples in focal loss
            alpha: balance positive and negative samples [0, 1] (increasing
                alpha increase weight of foreground classes (better recall))
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

    def comp_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss with subclass loss function

        Args:
            logits: logits for all foreground classes [*, C]
                * are arbitrary spatial dimensions, C is the number of
                foreground classes
            target: target classes. 0 is treated as background, >0 are
                treated as foreground classes. [*] where * are arbitrary
                spatial dimensions

        Returns:
            torch.Tensor: loss
        """
        return asymmetric_focal_loss_with_logits(
            logits,
            targets,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=self.reduction,
        )
