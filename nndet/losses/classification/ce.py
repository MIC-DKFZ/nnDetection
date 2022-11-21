# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
from torch.cuda.amp import autocast

from nndet.losses.ops import Loss, one_hot_smooth_last


class BCEWithLogitsLossOneHot(Loss, torch.nn.BCEWithLogitsLoss):
    def __init__(
        self,
        *args,
        smoothing: float = 0.0,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ):
        """
        BCE loss with one hot encoding of targets

        Args:
            num_classes: number of classes
            smoothing:  label smoothing
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32

        Warning:
            Only kept for backwards compatibility. Please don't use this class
            and use `nndet.losses.classification.bce.BinaryCrossEntropyLoss`
        """
        super().__init__(
            *args,
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
            **kwargs,
        )
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
            input: logits for all foreground classes [N, C]
                N is the number of anchors, and C is the number of foreground
                classes
            target: target classes. 0 is treated as background, >0 are
                treated as foreground classes. [N] is the number of anchors

        Returns:
            Tensor: final loss
        """
        num_classes = input.shape[1]
        target_one_hot = one_hot_smooth_last(
            target, num_classes=num_classes + 1, smoothing=self.smoothing
        )  # [N, C + 1]
        target_one_hot = target_one_hot[:, 1:]  # background is implicitly encoded

        if self.loss_fp32:
            with autocast(enabled=False):
                loss = self.loss_weight * super().forward(input.float(), target_one_hot.float())
        else:
            loss = self.loss_weight * super().forward(input, target_one_hot.to(dtype=input.dtype))
        return loss


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(
        self,
        *args,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ) -> None:
        """
        Same as CE from pytorch
        Targets can be float or long, it is castet to the correct type

        Args:
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
        """
        super().__init__(
            *args,
            **kwargs,
        )
        self.loss_weight = loss_weight
        self.loss_fp32 = loss_fp32
        if loss_fp32:
            logger.info(f"{self.__class__.__name__} uses FP32 loss computation.")

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Same as CE from pytorch
        """
        if self.loss_fp32:
            with autocast(enabled=False):
                loss = self.loss_weight * super().forward(input.float(), target.long())
        else:
            loss = self.loss_weight * super().forward(input, target.long())
        return loss


class BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def __init__(
        self,
        *args,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ) -> None:
        """
        Same as BCE with Logits from pytorch
        Targets can be float or long, it is castet to the correct type

        Args:
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
        """
        super().__init__(
            *args,
            **kwargs,
        )
        self.loss_weight = loss_weight
        self.loss_fp32 = loss_fp32
        if loss_fp32:
            logger.info(f"{self.__class__.__name__} uses FP32 loss computation.")

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Same as BCE with Logits from pytorch
        """
        if self.loss_fp32:
            with autocast(enabled=False):
                loss = self.loss_weight * super().forward(input.float(), target.float())
        else:
            loss = self.loss_weight * super().forward(input, target)
        return loss
