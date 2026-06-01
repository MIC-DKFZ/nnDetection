# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import torch

from nndet.losses.classification.functional.poly1 import (
    poly1_bce_with_logits,
    poly1_focal_loss_with_logits,
)
from nndet.losses.ops import SigmoidBaseLoss


class Poly1BFocalLoss(SigmoidBaseLoss):
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
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss with subclass loss function

        Args:
            preds: predictions (pre act) with shape [*, C], where
                * are arbitrary spatial dimensions, C is the number of
                *foreground* classes
            targets: target classes. 0 is treated as background, >0 are
                treated as foreground classes. [*] where * are arbitrary
                spatial dimensions

        Returns:
            torch.Tensor: loss
        """
        return poly1_focal_loss_with_logits(
            preds,
            targets,
            gamma=self.gamma,
            alpha=self.alpha,
            epsilon=self.epsilon,
            reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        return (
            f"alpha={self.alpha}, "
            f"gamma={self.gamma}, "
            f"epsilon={self.epsilon}, "
            f"smoothing={self.smoothing}"
            f"loss_weight={self.loss_weight}, "
            f"loss_fp32={self.loss_fp32}, "
            f"reduction={self.reduction}"
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
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss with subclass loss function

        Args:
            preds: predictions (pre act) with shape [*, C], where
                * are arbitrary spatial dimensions, C is the number of
                *foreground* classes
            targets: target classes. 0 is treated as background, >0 are
                treated as foreground classes. [*] where * are arbitrary
                spatial dimensions

        Returns:
            torch.Tensor: loss
        """
        return poly1_bce_with_logits(
            preds,
            targets,
            alpha=self.alpha,
            reduction=self.reduction,
            epsilon=self.epsilon,
        )

    def extra_repr(self) -> str:
        return (
            f"alpha={self.alpha}, "
            f"epsilon={self.epsilon}, "
            f"smoothing={self.smoothing}"
            f"loss_weight={self.loss_weight}, "
            f"loss_fp32={self.loss_fp32}, "
            f"reduction={self.reduction}"
        )
