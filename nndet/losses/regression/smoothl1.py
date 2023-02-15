# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.cuda.amp import autocast

from nndet.losses.ops import Loss


class SmoothL1Loss(Loss):
    def __init__(
        self,
        beta: float,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        reduction: str = "none",
    ):
        """
        Module wrapper for functional

        Args:
            beta: L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.

        See Also:
            :func:`smooth_l1_loss`
        """
        super().__init__(
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
            reduction=reduction,
        )
        self.beta = beta

    def forward(
        self,
        inp: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss

        Args:
            inp: predicted tensor (same shape as target)
            target: target tensor

        Returns:
            Tensor: computed loss
        """
        if self.loss_fp32:
            with autocast(enabled=False):
                loss = self.loss_weight * torch.nn.functional.smooth_l1_loss(
                    inp.float(),
                    target.float(),
                    beta=self.beta,
                    reduction=self.reduction,
                )
        else:
            loss = self.loss_weight * torch.nn.functional.smooth_l1_loss(
                inp,
                target,
                beta=self.beta,
                reduction=self.reduction,
            )
        return loss
