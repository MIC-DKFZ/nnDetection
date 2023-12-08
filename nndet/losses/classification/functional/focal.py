# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.nn import functional as F

from nndet.losses.ops import reduction_helper


@torch.compile
def focal_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float,
    alpha: float = -1,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Focal loss
    https://arxiv.org/abs/1708.02002

    Args:
        logits: predicted logits [*]
        target: binary targets [*]
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
        :class:`BFocalLoss`
    """
    p = torch.sigmoid(logits)
    focal_term = (1.0 - (p * target + (1 - p) * (1 - target))) ** float(gamma)
    loss = focal_term * F.binary_cross_entropy_with_logits(logits, target, reduction="none")

    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss
    return reduction_helper(loss, reduction=reduction)
