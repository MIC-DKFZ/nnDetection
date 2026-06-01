# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.cuda.amp import autocast

import nndet.core.ops_torch as ops_torch
from nndet.losses.ops import reduction_helper


@autocast(enabled=False)
def generalized_box_iou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    reduction: str,
    eps: float = 0,
) -> torch.Tensor:
    """
    Distance IoU Loss
    L = 1 - IoU + d^2(c, c_gt) / diag_enclosing^2

    Args:
        pred_boxes: predicted boxes [N, dims] (x1, y1, x2, y2, z1, z2)
        target_boxes: target boxes [N, dims] (x1, y1, x2, y2, z1, z2)
        eps: small constant for numerical stability
        reduction: 'mean'|'sum'|'none'
            mean: mean of loss over entire batch
            sum: sum of loss over entire batch
            none: no reduction

    Returns:
        torch.Tensor: computed loss

    Notes:
        Need to compute IoU in float32 (autocast=False) because the
        volume/area can be to large
    """
    loss = ops_torch.generalized_box_iou_paired(
        boxes1=pred_boxes,
        boxes2=target_boxes,
        eps=eps,
    )
    return reduction_helper(loss, reduction=reduction)
