from typing import Optional

import torch
from loguru import logger
from torch.cuda.amp import autocast

from nndet.losses.ops import Loss, one_hot_smooth_last, reduction_helper


class BCELoss(Loss):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        smoothing: float = 0.0,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        reduction: str = "sum",
    ):
        """
        BCE loss with one hot encoding of targets

        Args:
            weight: equivalent to weight parameter of BCE loss of pytorch
            smoothing:  label smoothing
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            reduction: 'mean'|'sum'|'none' |'mean_last_sum'
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
        self.weight = weight
        self.smoothing = smoothing
        if smoothing > 0:
            logger.info(f"Running label smoothing with smoothing: {smoothing}")

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute bce loss based on one hot encoding

        Args:
            input: logits for all foreground classes [*, C]
                * are arbitrary spatial dimensions, C is the number of
                foreground classes
            target: target classes. 0 is treated as background, >0 are
                treated as foreground classes. [*] where * are arbitrary
                spatial dimensions

        Returns:
            Tensor: computed loss
        """
        num_classes = input.shape[-1]
        target_one_hot = one_hot_smooth_last(
            target, num_classes=num_classes + 1, smoothing=self.smoothing
        )  # [N, *, C + 1]
        target_one_hot = target_one_hot[..., 1:]  # background is implicitly encoded

        if self.loss_fp32:
            with autocast(enabled=False):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    input.float(),
                    target_one_hot.float(),
                    reduction="none",
                    weight=self.weight,
                )
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                input,
                target_one_hot.to(dtype=input.dtype),
                reduction="none",
                weight=self.weight,
            )
        return self.loss_weight * reduction_helper(loss, reduction=self.reduction)


class CELoss(Loss):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        reduction: str = "sum",
    ):
        """
        CE loss

        Args:
            weight: equivalent to weight parameter of CE loss of pytorch
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
        self.weight = weight

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss

        Args:
            input: logits for all foreground classes [*, C]
                * are arbitrary spatial dimensions, C is the number of
                foreground classes
            target: target classes. 0 is treated as background, >0 are
                treated as foreground classes. [*] where * are arbitrary
                spatial dimensions

        Returns:
            Tensor: computed loss

        Warning:
            Note the ordering of the input is different from pytorch!
        """
        permute_inputs = input.ndim > 2
        if permute_inputs:
            # permute class channel to first axis
            _input = input.movedim(-1, 1)
        else:
            _input = input

        if self.loss_fp32:
            with autocast(enabled=False):
                loss = self.loss_weight * torch.nn.functional.cross_entropy(
                    _input.float(),
                    target.long(),
                    weight=self.weight,
                    reduction=self.reduction,
                )
        else:
            loss = self.loss_weight * torch.nn.functional.cross_entropy(
                _input,
                target.long(),
                weight=self.weight,
                reduction=self.reduction,
            )

        if permute_inputs and self.reduction.lower() == "none":
            # restore permutation
            loss = loss.movedim(1, -1)
        return loss
