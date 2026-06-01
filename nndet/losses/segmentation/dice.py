# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.cuda.amp import autocast

from nndet.losses.mask.dice import BDiceMaskLoss
from nndet.losses.mask.functional.dice import soft_dice
from nndet.losses.ops import Loss, one_hot_smooth_first, reduction_helper


class DiceSegLoss(Loss):
    def __init__(
        self,
        batch_dice: bool = False,
        do_bg: bool = False,
        smoothing: float = 0.0,
        loss_weight: float = 1,
        loss_fp32: bool = False,
        smooth_nom: float = 1e-5,
        smooth_denom: float = 1e-5,
        reduction: str = "mean",
    ) -> None:
        """
        Compute Dice loss (softmax based)

        Args:
            batch_dice: compute statistics for each class across the whole batch
                instead of computing if per image per class. Defaults to False.
            do_bg: compute loss for background
            smoothing: apply label smoothing to targets. Label smoothing is
                somewhat experimental here, use on your own risk!
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            smooth_nom: constant added to nominator for numerical stability.
                Defaults to 1e-5.
            smooth_denom: constant added to denominator for numerical stability.
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
        self.smoothing = smoothing
        self.do_bg = do_bg

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Loss

        Args:
            preds: predictions (without act). [N, C, *], where N is the batch
                size, C is the number of classes, * are arbitrary spatial
                dimensions
            targets: numerical target values. [N, *], where N is the batch
                size, * are arbitrary spatial dimensions

        Returns:
            torch.Tensor: computed loss
        """
        num_classes = preds.shape[1]
        targets_one_hot = one_hot_smooth_first(
            targets,
            num_classes=num_classes,
            smoothing=self.smoothing,
        )

        if self.loss_fp32:
            with autocast(enabled=False):
                probs = torch.nn.functional.softmax(preds.float(), dim=1)
                if not self.do_bg:
                    probs = probs[:, 1:]
                    targets_one_hot = targets_one_hot[:, 1:]
                loss = soft_dice(
                    preds=probs,
                    targets_one_hot=targets_one_hot.float(),
                    smooth_nom=self.smooth_nom,
                    smooth_denom=self.smooth_denom,
                    batch_dice=self.batch_dice,
                )  # [N, C]
        else:
            probs = torch.nn.functional.softmax(preds, dim=1)
            if not self.do_bg:
                probs = probs[:, 1:]
                targets_one_hot = targets_one_hot[:, 1:]
            loss = soft_dice(
                preds=probs,
                targets_one_hot=targets_one_hot,
                smooth_nom=self.smooth_nom,
                smooth_denom=self.smooth_denom,
                batch_dice=self.batch_dice,
            )  # [N, C]

        return self.loss_weight * reduction_helper(loss, reduction=self.reduction)

    def extra_repr(self) -> str:
        return (
            f"do_bg={self.do_bg}, "
            f"batch_dice={self.batch_dice}, "
            f"smooth_nom={self.smooth_nom}, "
            f"smooth_denom={self.smooth_denom}, "
            f"smoothing={self.smoothing}, "
            f"loss_weight={self.loss_weight}, "
            f"loss_fp32={self.loss_fp32}, "
            f"reduction={self.reduction}"
        )


class BDiceSegLoss(BDiceMaskLoss):
    def __init__(
        self,
        batch_dice: bool = False,
        do_bg: bool = False,
        smoothing: float = 0.0,
        loss_weight: float = 1,
        loss_fp32: bool = False,
        smooth_nom: float = 1e-5,
        smooth_denom: float = 1e-5,
        reduction: str = "mean",
    ) -> None:
        """
        Compute Dice loss (sigmoid based)

        Args:
            batch_dice: compute statistics for each class across the whole batch
                instead of computing if per image per class. Defaults to False.
            do_bg: compute loss for background
            smoothing: apply label smoothing to targets. Label smoothing is
                somewhat experimental here, use on your own risk!
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            smooth_nom: constant added to nominator for numerical stability.
                Defaults to 1e-5.
            smooth_denom: constant added to denominator for numerical stability.
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
            batch_dice=batch_dice,
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
            smooth_nom=smooth_nom,
            smooth_denom=smooth_denom,
            reduction=reduction,
        )
        self.smoothing = smoothing
        self.do_bg = do_bg

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Loss

        Args:
            preds: predictions (without act). [N, C, *], where N is the batch
                size, C is the number of classes, * are arbitrary spatial
                dimensions
            targets: numerical target values. [N, *], where N is the batch
                size, * are arbitrary spatial dimensions

        Returns:
            torch.Tensor: computed loss
        """
        targets_one_hot = one_hot_smooth_first(
            targets,
            num_classes=preds.shape[1],
            smoothing=self.smoothing,
        )
        if not self.do_bg:
            _preds = preds[:, 1:]
            _targets_one_hot = targets_one_hot[:, 1:]
        else:
            _preds = preds
            _targets_one_hot = targets_one_hot
        return super().forward(
            preds=_preds,
            targets=_targets_one_hot,
        )

    def extra_repr(self) -> str:
        return (
            f"do_bg={self.do_bg}, "
            f"batch_dice={self.batch_dice}, "
            f"smooth_nom={self.smooth_nom}, "
            f"smooth_denom={self.smooth_denom}, "
            f"smoothing={self.smoothing}, "
            f"loss_weight={self.loss_weight}, "
            f"loss_fp32={self.loss_fp32}, "
            f"reduction={self.reduction}"
        )
