# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
from torch.cuda.amp import autocast

from nndet.losses.ops import SigmoidBaseLoss, TorchLoss, reduction_helper


class BCELoss(SigmoidBaseLoss):
    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        smoothing: float = 0.0,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        reduction: str = "sum",
        weight: Optional[torch.Tensor] = None,
    ):
        """
        BCE loss with one hot encoding of targets (loss is only computed
        on foreground classes!)

        Args:
            pos_weight: equivalent to pos_weight parameter of BCE loss of
                pytorch (weights positive class)
            smoothing:  label smoothing
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            reduction: 'mean'|'sum'|'none' |'mean_last_sum'
                mean: mean of loss over entire batch
                sum: sum of loss over entire batch
                none: no reduction
                mean_last_sum: mean over last dimension, sum across others
            weight: equivalent to weight parameter of BCE loss of pytorch
                (weights batch elements)
        """
        super().__init__(
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
            reduction=reduction,
            smoothing=smoothing,
        )
        self.register_buffer("pos_weight", pos_weight)
        self.pos_weight: Optional[torch.Tensor]
        self.register_buffer("weight", weight)
        self.weight: Optional[torch.Tensor]

    @property
    def reduction(self) -> str:
        if self.torch_reduction.lower() == "mean":
            return self.torch_reduction
        else:
            return self.helper_reduction

    @reduction.setter
    def reduction(self, key: str):
        _key = key.lower()
        if _key == "mean":
            self.torch_reduction = "mean"
            self.helper_reduction = "none"
        else:
            self.torch_reduction = "none"
            self.helper_reduction = _key

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
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            preds,
            targets,
            reduction=self.torch_reduction,
            weight=self.weight,
            pos_weight=self.pos_weight,
        )
        return reduction_helper(loss, reduction=self.helper_reduction)

    def extra_repr(self) -> str:
        return (
            f"weight={self.weight}, "
            f"smoothing={self.smoothing}, "
            f"loss_weight={self.loss_weight}, "
            f"loss_fp32={self.loss_fp32}, "
            f"reduction={self.reduction}"
        )


class CELoss(TorchLoss):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        smoothing: float = 0.0,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        reduction: str = "sum",
    ):
        """
        CE loss

        Args:
            weight: equivalent to weight parameter of CE loss of pytorch
            smoothing:  label smoothing
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            reduction: 'mean'|'sum'|'none'|'mean_last_sum'
                mean: mean of loss over entire batch
                sum: sum of loss over entire batch
                none: no reduction
                mean_last_sum: mean over last dimension, sum across others
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
        Compute loss

        Args:
            preds: predictions (pre act) with shape [*, C], where
                * are arbitrary spatial dimensions, C is the number of
                *foreground* classes
            targets: target classes. 0 is treated as background, >0 are
                treated as foreground classes. [*] where * are arbitrary
                spatial dimensions

        Returns:
            Tensor: computed loss

        Warning:
            Note the ordering of the input is different from pytorch!
        """
        permute_inputs = preds.ndim > 2
        if permute_inputs:
            # permute class channel to first axis
            _input = preds.movedim(-1, 1)
        else:
            _input = preds

        if self.loss_fp32:
            with autocast(enabled=False):
                loss = torch.nn.functional.cross_entropy(
                    _input.float(),
                    targets.long(),
                    weight=self.weight,
                    reduction=self.torch_reduction,
                    label_smoothing=self.smoothing,
                )
        else:
            loss = torch.nn.functional.cross_entropy(
                _input,
                targets.long(),
                weight=self.weight,
                reduction=self.torch_reduction,
                label_smoothing=self.smoothing,
            )

        if permute_inputs and self.torch_reduction.lower() == "none":
            # restore permutation
            loss = loss.movedim(1, -1)
        return self.loss_weight * reduction_helper(loss, reduction=self.helper_reduction)

    def extra_repr(self) -> str:
        return (
            f"weight={self.weight}, "
            f"smoothing={self.smoothing}, "
            f"loss_weight={self.loss_weight}, "
            f"loss_fp32={self.loss_fp32}, "
            f"reduction={self.reduction}"
        )
