# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
from torch.cuda.amp import autocast

from nndet.losses.ops import TorchLoss, one_hot_smooth_first, reduction_helper


class CESegLoss(TorchLoss):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        smoothing: float = 0.0,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        reduction: str = "mean",
    ) -> None:
        """
        Wrapper for PyTorch CE Loss. Targets will always be casted to long
        before calling the loss function!

        Args:
            weight: weight for CE loss, see PyTorch docs for more info.
            smoothing: Apply label smoothing to loss. See PyTorch docs for
                more info.
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            reduction: reduction of loss. Refer to
                `nndet.losses.ops.reduction_helper` for all available options.
        """
        super().__init__(
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
            reduction=reduction,
        )
        self.register_buffer("weight", weight)
        self.weight: Optional[torch.Tensor]
        self.smoothing = smoothing

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
        _fn = torch.nn.functional.cross_entropy
        if self.loss_fp32:
            with autocast(enabled=False):
                loss = _fn(
                    preds.float(),
                    targets.long(),
                    label_smoothing=self.smoothing,
                    weight=self.weight,
                    reduction=self.torch_reduction,
                )
        else:
            loss = _fn(
                preds,
                targets.long(),
                label_smoothing=self.smoothing,
                weight=self.weight,
                reduction=self.torch_reduction,
            )
        return self.loss_weight * reduction_helper(loss, reduction=self.helper_reduction)

    def extra_repr(self) -> str:
        return (
            f"weight={self.weight}, "
            f"smoothing={self.smoothing}, "
            f"loss_weight={self.loss_weight}, "
            f"loss_fp32={self.loss_fp32}, "
            f"reduction={self.reduction}"
        )


class BCESegLoss(TorchLoss):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        do_bg: bool = False,
        smoothing: float = 0.0,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        reduction: str = "mean",
    ) -> None:
        """
        Wrapper for PyTorch BCE Loss. Targets will always be casted to long
        before calling the loss function!

        Args:
            weight: weiught for BCE loss, see PyTorch docs for more info.
            do_bg: compute loss for background
            smoothing: Apply label smoothing to loss.
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            reduction: reduction of loss. Refer to
                `nndet.losses.ops.reduction_helper` for all available options.
        """
        super().__init__(
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
            reduction=reduction,
        )
        self.register_buffer("weight", weight)
        self.weight: Optional[torch.Tensor]
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

        _fn = torch.nn.functional.binary_cross_entropy_with_logits
        if self.loss_fp32:
            with autocast(enabled=False):
                loss = _fn(
                    _preds.float(),
                    _targets_one_hot.float(),
                    weight=self.weight,
                    reduction=self.torch_reduction,
                )
        else:
            loss = _fn(
                _preds,
                _targets_one_hot.float(),
                weight=self.weight,
                reduction=self.torch_reduction,
            )
        return self.loss_weight * reduction_helper(loss, reduction=self.helper_reduction)

    def extra_repr(self) -> str:
        return (
            f"weight={self.weight}, "
            f"do_bg={self.do_bg}, "
            f"smoothing={self.smoothing}, "
            f"loss_weight={self.loss_weight}, "
            f"loss_fp32={self.loss_fp32}, "
            f"reduction={self.reduction}"
        )
