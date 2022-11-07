# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.nn import functional as F

from nndet.losses.ops import reduction_helper


def asymmetric_focal_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float,
    alpha: float = -1,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Asymmetric Focal loss
    Inspired by https://arxiv.org/abs/2008.13367
    and https://arxiv.org/abs/1907.10982 (without margin)

    Args:
        logits: predicted logits [N, dims]
        target: binary targets [N, dims]
        gamma: balance easy and hard examples in focal loss
        alpha: balance factor for background (different from focal loss)
        reduction: 'mean'|'sum'|'none'
            mean: mean of loss over entire batch
            sum: sum of loss over entire batch
            none: no reduction

    Returns:
        torch.Tensor: loss

    See Also
        :class:`BFocalLossWithLogits`, :class:`FocalLossWithLogits`
    """
    p = torch.sigmoid(logits)
    focal_term = (1 - (1 - p) * (1 - target)) ** float(gamma)
    loss = focal_term * F.binary_cross_entropy_with_logits(logits, target, reduction="none")

    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss
    return reduction_helper(loss, reduction=reduction)


asymmetric_focal_loss_with_logits_jit: torch.jit.ScriptFunction = torch.jit.script(asymmetric_focal_loss_with_logits)
