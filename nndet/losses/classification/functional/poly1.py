# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.nn import functional as F

from nndet.losses.ops import reduction_helper


def poly1_focal_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float,
    alpha: float = -1,
    reduction: str = "mean",
    epsilon: float = -1,
) -> torch.Tensor:
    """
    Focal loss
    https://arxiv.org/abs/1708.02002
    Poly1 Focal-Loss
    https://openreview.net/forum?id=gSdSJoenupI

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
        epsilon: epsilon of poly term.

    Returns:
        torch.Tensor: loss

    See Also
        :class:`BFocalLossWithLogits`, :class:`FocalLossWithLogits`
    """
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    focal_term = (1.0 - pt) ** float(gamma)
    poly_term = (1.0 - pt) ** float(gamma + 1)
    ce_term = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

    loss = focal_term * ce_term + epsilon * poly_term

    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss
    return reduction_helper(loss, reduction=reduction)


def poly1_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = -1,
    reduction: str = "sum",
    epsilon: float = -1,
) -> torch.Tensor:
    """
    Poly1 BCE
    https://openreview.net/forum?id=gSdSJoenupI

    Args:
        logits: predicted logits [N, dims]
        target: binary targets [N, dims]
        alpha: balance positive and negative samples [0, 1] (increasing
            alpha increase weight of foreground classes (better recall))
        reduction: 'mean'|'sum'|'none'
            mean: mean of loss over entire batch
            sum: sum of loss over entire batch
            none: no reduction
        epsilon: epsilon of poly term.

    Returns:
        torch.Tensor: loss
    """
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    poly_term = 1.0 - pt
    ce_term = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = ce_term + epsilon * poly_term

    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss
    return reduction_helper(loss, reduction=reduction)
