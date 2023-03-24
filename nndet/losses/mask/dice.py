# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.cuda.amp import autocast

from nndet.losses.mask.functional.dice import soft_dice
from nndet.losses.ops import Loss, reduction_helper


class BDiceMaskLoss(Loss):
    def __init__(
        self,
        batch_dice: bool = False,
        loss_weight: float = 1,
        loss_fp32: bool = False,
        smooth_nom: float = 0,
        smooth_denom: float = 1e-5,
        reduction: str = "mean",
    ) -> None:
        """
        Compute Dice loss for binary masks

        Args:
            batch_dice: compute statistics for each class across the whole batch
                instead of computing if per image per class. Defaults to False.
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            smooth_nom: constant added to nominator for numerical stabilty.
                Defaults to 0.
            smooth_denom: contant added to denominator for numerical stabilty.
                Defaults to 1e-5.
            reduction: reduction of loss. Refer to
                `nndet.losses.ops.reduction_helper` for all available options.
                Reduction 'none' is not supported here. If batch dice
                is active, only 'sum' and 'mean' are supported.

        Raises:
            ValueError: reduction 'none' is not supported
            ValueError: if batch dice is active, only 'sum' and 'mean'
                reduction are supported
        """
        super().__init__(
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
            reduction=reduction,
        )
        if self.reduction == "none":
            raise ValueError("Reduction 'none' is not supported for dice loss")
        if batch_dice and self.reduction not in ["sum", "mean"]:
            raise ValueError("If batch dice is active, only 'sum' and 'mean' reduction are supported.")
        self.batch_dice = batch_dice
        self.smooth_nom = smooth_nom
        self.smooth_denom = smooth_denom

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Loss

        Args:
            preds: predictions (pre act). [N, C, *], where N is the batch
                size, C is the number of classes, * are arbitrary spatial
                dimensions
            targets: targets encoded as one hot. [N, C, *], where
                N is the batch size, C is the number of classes, * are
                arbitrary spatial dimensions

        Returns:
            torch.Tensor: computed loss
        """
        if self.loss_fp32:
            with autocast(enabled=False):
                probs = torch.nn.functional.sigmoid(preds.float())
                loss = soft_dice(
                    preds=probs,
                    targets_one_hot=targets.float(),
                    smooth_nom=self.smooth_nom,
                    smooth_denom=self.smooth_denom,
                    batch_dice=self.batch_dice,
                )  # [N, C]
        else:
            probs = torch.nn.functional.sigmoid(preds)
            loss = soft_dice(
                preds=probs,
                targets_one_hot=targets,
                smooth_nom=self.smooth_nom,
                smooth_denom=self.smooth_denom,
                batch_dice=self.batch_dice,
            )  # [N, C]
        return self.loss_weight * reduction_helper(loss, reduction=self.reduction)

    def extra_repr(self) -> str:
        return (
            f"batch_dice={self.batch_dice}, "
            f"smooth_nom={self.smooth_nom}, "
            f"smooth_denom={self.smooth_denom}, "
            f"loss_weight={self.loss_weight}, "
            f"loss_fp32={self.loss_fp32}, "
            f"reduction={self.reduction}"
        )
