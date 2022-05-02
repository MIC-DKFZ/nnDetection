"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from abc import abstractmethod

import torch
from loguru import logger
from torch.cuda.amp import autocast


class Loss(torch.nn.Module):
    def __init__(
        self,
        *args,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        reduction: str = "sum",
        **kwargs,
    ) -> None:
        """
        Base class for all nnDetection losses

        Args:
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            reduction: 'mean'|'sum'|'none'
                mean: mean of loss over entire batch
                sum: sum of loss over entire batch
                none: no reduction
        """
        super().__init__(*args, **kwargs)
        self.reduction = reduction
        self.loss_fp32 = loss_fp32
        self.loss_weight = loss_weight

    @property
    def loss_fp32(self) -> bool:
        return self._loss_fp32

    @loss_fp32.setter
    def loss_fp32(self, val: bool):
        self._loss_fp32 = val
        if val:
            logger.info(f"{self.__class__.__name__} uses FP32 loss computation.")


class SigmoidBaseLoss(Loss):
    def __init__(
        self,
        loss_weight: float = 1,
        loss_fp32: bool = False,
        reduction: str = "sum",
        smoothing: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
            reduction=reduction,
            **kwargs,
        )
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
            targets,
            num_classes=num_classes,
            smoothing=self.smoothing,
        )
        target_onehot = target_onehot[:, 1:]

        if self.loss_fp32:
            with autocast(enabled=False):
                loss = self.comp_loss(
                    logits=logits.float(),
                    targets=target_onehot.float(),
                )
        else:
            loss = self.comp_loss(
                logits=logits,
                targets=target_onehot.to(dtype=logits.dtype),
            )
        return self.loss_weight * loss

    @abstractmethod
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
        raise NotImplementedError


def reduction_helper(
    data: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """
    Helper to collapse data with different modes

    Args:
        data: data to collapse
        reduction: type of reduction. One of `mean`, `sum`, 'none'

    Returns:
        Tensor: reduced data
    """
    if reduction.lower() == "mean":
        return torch.mean(data)
    if reduction.lower() == "none":
        return data
    if reduction.lower() == "sum":
        return torch.sum(data)
    raise AttributeError("Reduction parameter unknown.")


def one_hot_smooth_last(
    data: torch.Tensor,
    num_classes: int,
    smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Convert data with numbers into one-hot-encoding. The classes are
    added at the end of the tensor!

    Args:
        data: input data with numbers [dims]
        num_classes: number of classes
        smoothing: Optional smoothing factor. Defaults to 0.0.

    Returns:
        torch.Tensor: one-hot-encoded targets. [dims, num_classes]
    """
    targets = (
        torch.empty(size=(*data.shape, num_classes), device=data.device)
        .fill_(smoothing / num_classes)
        .scatter_(-1, data.long().unsqueeze(-1), 1.0 - smoothing)
    )
    return targets


def ont_hot_smooth_first(
    data: torch.Tensor,
    num_classes: int,
    smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Convert data with numbers into one-hot-encoding. The classes are
    added in the first dimension of the tensor!

    Args:
        data: input data with numbers [dims]
        num_classes: number of classes
        smoothing: Optional smoothing factor. Defaults to 0.0.

    Returns:
        torch.Tensor: one-hot-encoded targets. [dims[0], num_classes, other_dims]
    """
    shape = data.shape
    targets = (
        torch.empty(size=(shape[0], num_classes, *shape[1:]), device=data.device)
        .fill_(smoothing / num_classes)
        .scatter_(1, data.long().unsqueeze(1), 1.0 - smoothing)
    )
    return targets
