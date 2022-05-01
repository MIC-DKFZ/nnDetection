import torch
from loguru import logger
from torch.cuda.amp import autocast

from nndet.losses.classification.functional.asymfocal import (
    asymmetric_focal_loss_with_logits_jit as asymmetric_focal_loss_with_logits,
)
from nndet.losses.classification.functional.focal import (
    focal_loss_with_logits_jit as focal_loss_with_logits,
)
from nndet.losses.ops import Loss, ont_hot_smooth_first


class FocalLossWithLogits(Loss):
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
        )
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing
        if smoothing > 0:
            logger.info(f"Running label smoothing with smoothing: {smoothing}")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss

        Args:
            logits: predicted logits [N, C, dims], where N is the batch size,
                C number of classes, dims are arbitrary spatial dimensions
                (background classes should be located at channel 0 if
                ignore background is enabled)
            targets: targets encoded as numbers [N, dims], where N is the
                batch size, dims are arbitrary spatial dimensions

        Returns:
            torch.Tensor: loss
        """
        num_classes = logits.shape[1] + 1
        target_onehot = ont_hot_smooth_first(
            targets, num_classes=num_classes, smoothing=self.smoothing
        )
        target_onehot = target_onehot[:, 1:]

        if self.loss_fp32:
            with autocast(enabled=False):
                loss = self.loss_weight * focal_loss_with_logits(
                    logits.float(),
                    target_onehot.float(),
                    gamma=self.gamma,
                    alpha=self.alpha,
                    reduction=self.reduction,
                )
        else:
            loss = self.loss_weight * focal_loss_with_logits(
                logits,
                target_onehot.to(dtype=logits.dtype),
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=self.reduction,
            )
        return loss


class AsymmetricFocalLossWithLogits(Loss):
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
        )
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing
        if smoothing > 0:
            logger.info(f"Running label smoothing with smoothing: {smoothing}")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss

        Args:
            logits: predicted logits [N, C, dims], where N is the batch size,
                C number of classes, dims are arbitrary spatial dimensions
                (background classes should be located at channel 0 if
                ignore background is enabled)
            targets: targets encoded as numbers [N, dims], where N is the
                batch size, dims are arbitrary spatial dimensions

        Returns:
            torch.Tensor: loss
        """
        num_classes = logits.shape[1] + 1
        target_onehot = ont_hot_smooth_first(
            targets, num_classes=num_classes, smoothing=self.smoothing
        )
        target_onehot = target_onehot[:, 1:]

        if self.loss_fp32:
            with autocast(enabled=False):
                loss = self.loss_weight * asymmetric_focal_loss_with_logits(
                    logits.float(),
                    target_onehot.float(),
                    gamma=self.gamma,
                    alpha=self.alpha,
                    reduction=self.reduction,
                )
        else:
            loss = self.loss_weight * asymmetric_focal_loss_with_logits(
                logits,
                target_onehot.to(dtype=logits.dtype),
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=self.reduction,
            )
        return loss
