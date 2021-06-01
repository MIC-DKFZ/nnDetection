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

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch import Tensor
from loguru import logger

from nndet.losses.base import reduction_helper
from nndet.utils import make_onehot_batch


def one_hot_smooth(data,
                   num_classes: int,
                   smoothing: float = 0.0,
                   ):
    targets = torch.empty(size=(*data.shape, num_classes), device=data.device)\
        .fill_(smoothing / num_classes)\
        .scatter_(-1, data.long().unsqueeze(-1), 1. - smoothing)
    return targets


@torch.jit.script
def focal_loss_with_logits(
        logits: torch.Tensor,
        target: torch.Tensor, gamma: float,
        alpha: float = -1,
        reduction: str = "mean",
        ) -> torch.Tensor:
    """
    Focal loss
    https://arxiv.org/abs/1708.02002

    Args:
        logits: predicted logits [N, dims]
        target: (float) binary targets [N, dims]
        gamma: balance easy and hard examples in focal loss
        alpha: balance positive and negative samples [0, 1] (increasing
            alpha increase weight of foreground classes (better recall))
        reduction: 'mean'|'sum'|'none'
            mean: mean of loss over entire batch
            sum: sum of loss over entire batch
            none: no reduction

    Returns:
        torch.Tensor: loss

    See Also
        :class:`BFocalLossWithLogits`, :class:`FocalLossWithLogits`
    """
    bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')

    p = torch.sigmoid(logits)
    pt = (p * target + (1 - p) * (1 - target))

    focal_term = (1. - pt).pow(gamma)
    loss = focal_term * bce_loss

    if alpha >= 0:
        alpha_t = (alpha * target + (1 - alpha) * (1 - target))
        loss = alpha_t * loss

    return reduction_helper(loss, reduction=reduction)


class FocalLossWithLogits(nn.Module):
    def __init__(self,
                 gamma: float = 2,
                 alpha: float = -1,
                 reduction: str = "sum",
                 loss_weight: float = 1.,
                 ):
        """
        Focal loss with multiple classes (uses one hot encoding and sigmoid)

        Args:
            gamma: balance easy and hard examples in focal loss
            alpha: balance positive and negative samples [0, 1] (increasing
                alpha increase weight of foreground classes (better recall))
            reduction: 'mean'|'sum'|'none'
                mean: mean of loss over entire batch
                sum: sum of loss over entire batch
                none: no reduction
        loss_weight: scalar to balance multiple losses
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
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
        n_classes = logits.shape[1] + 1
        target_onehot = make_onehot_batch(targets, n_classes=n_classes).float()
        target_onehot = target_onehot[:, 1:]

        return self.loss_weight * focal_loss_with_logits(
            logits, target_onehot,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=self.reduction,
            )


class BCEWithLogitsLossOneHot(torch.nn.BCEWithLogitsLoss):
    def __init__(self,
                 *args,
                 num_classes: int,
                 smoothing: float = 0.0,
                 loss_weight: float = 1.,
                 **kwargs,
                 ):
        """
        BCE loss with one hot encoding of targets

        Args:
            num_classes: number of classes
            smoothing:  label smoothing
            loss_weight: scalar to balance multiple losses
        """
        super().__init__(*args, **kwargs)
        self.smoothing = smoothing
        if smoothing > 0:
            logger.info(f"Running label smoothing with smoothing: {smoothing}")
        self.num_classes = num_classes
        self.loss_weight = loss_weight

    def forward(self,
                input: Tensor,
                target: Tensor,
                ) -> Tensor:
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
        target_one_hot = one_hot_smooth(
            target, num_classes=self.num_classes + 1, smoothing=self.smoothing)  # [N, C + 1]
        target_one_hot = target_one_hot[:, 1:]  # background is implicitly encoded

        return self.loss_weight * super().forward(input, target_one_hot.float())


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self,
                 *args,
                 loss_weight: float = 1.,
                 **kwargs,
                 ) -> None:
        """
        Same as CE from pytorch with additional loss weight for uniform API
        """
        super().__init__(*args, **kwargs)
        self.loss_weight = loss_weight

    def forward(self,
                input: Tensor,
                target: Tensor,
                ) -> Tensor:
        """
        Same as CE from pytorch
        """
        return self.loss_weight * super().forward(input, target)
