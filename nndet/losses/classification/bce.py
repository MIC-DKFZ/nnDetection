# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

from nndet.losses.ops import SigmoidBaseLoss


class BinaryCrossEntropyLoss(SigmoidBaseLoss):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        loss_fp32: bool = False,
        loss_weight: float = 1.0,
        reduction: str = "sum",
        smoothing: float = 0.0,
    ):
        """
        BCE Loss with multiple classes (uses one hot encoding and sigmoid)

        Args:
            weight: weight tensor for BCE loss (see Torch docs)
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
        self.weight = weight

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
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            weight=self.weight,
            reduction=self.reduction,
        )
