from typing import Optional

import torch
from torch.cuda.amp import autocast

from nndet.losses.ops import Loss, one_hot_smooth_first, reduction_helper


class CESegLoss(Loss):
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
            weight: weiught for CE loss, see PyTorch docs for more info.
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
        self.weight = weight
        self.smoothing = smoothing

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss

        Args:
            input: predicted logits. Shape [N, C, *] where N is the batch size,
                C is the number of classes and * are arbitrary spatial
                dimensions
            target: numerical tensor specifying the labels of shape [N, *],
                where N is the batch size and * are arbitrary dimensions

        Returns:
            torch.Tensor: computed loss
        """
        _fn = torch.nn.functional.cross_entropy
        if self.loss_fp32:
            with autocast(enabled=False):
                loss = _fn(
                    input.float(),
                    target.long(),
                    label_smoothing=self.smoothing,
                    weight=self.weight,
                    reduction="none",
                )
        else:
            loss = _fn(
                input,
                target.long(),
                label_smoothing=self.smoothing,
                weight=self.weight,
                reduction="none",
            )
        return self.loss_weight * reduction_helper(loss, reduction=self.reduction)

    def extra_repr(self) -> str:
        return (
            f"weight={self.weight}"
            f"smoothing={self.smoothing} "
            f"loss_weight={self.loss_weight}, "
            f"loss_fp32={self.loss_fp32}, "
            f"reduction={self.reduction}, "
        )


class BCESegLoss(Loss):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
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
        self.weight = weight
        self.smoothing = smoothing

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss

        Args:
            input: predicted logits. Shape [N, C, *] where N is the batch size,
                C is the number of classes and * are arbitrary spatial
                dimensions
            target: numerical tensor specifying the labels of shape [N, *],
                where N is the batch size and * are arbitrary dimensions

        Returns:
            torch.Tensor: computed loss
        """
        num_classes = input.shape[1]
        _target = one_hot_smooth_first(
            target,
            num_classes=num_classes + 1,
            smoothing=self.smoothing,
        )
        _target = _target[:, 1:]

        _fn = torch.nn.functional.binary_cross_entropy_with_logits
        if self.loss_fp32:
            with autocast(enabled=False):
                loss = _fn(
                    input.float(),
                    _target.float(),
                    weight=self.weight,
                    reduction="none",
                )
        else:
            loss = _fn(
                input,
                _target.float(),
                weight=self.weight,
                reduction="none",
            )
        return self.loss_weight * reduction_helper(loss, reduction=self.reduction)

    def extra_repr(self) -> str:
        return (
            f"weight={self.weight}"
            f"smoothing={self.smoothing} "
            f"loss_weight={self.loss_weight}, "
            f"loss_fp32={self.loss_fp32}, "
            f"reduction={self.reduction}, "
        )
