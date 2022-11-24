# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
# SPDX-License-Identifier: Apache-2.0
#
# L1 loss from fvcore (https://github.com/facebookresearch/fvcore) licensed under
# SPDX-FileCopyrightText: 2019, Facebook, Inc
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

import torch


__all__ = ["SmoothL1Loss", "smooth_l1_loss"]

from nndet.core.boxes.ops import generalized_box_iou
from nndet.losses.base import reduction_helper


class SmoothL1Loss(torch.nn.Module):
    def __init__(self,
                 beta: float,
                 reduction: Optional[str] = None,
                 loss_weight: float = 1.,
                 ):
        """
        Module wrapper for functional

        Args:
            beta (float): L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            reduction (str): 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.

        See Also:
            :func:`smooth_l1_loss`
        """
        super().__init__()
        self.reduction = reduction
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss

        Args:
            inp (torch.Tensor): predicted tensor (same shape as target)
            target (torch.Tensor): target tensor

        Returns:
            Tensor: computed loss
        """
        return self.loss_weight * reduction_helper(smooth_l1_loss(inp, target, self.beta), self.reduction)


def smooth_l1_loss(inp, target, beta: float):
    """
    From https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py

    Smooth L1 loss defined in the Fast R-CNN paper as:
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    Smooth L1 loss is related to Huber loss, which is defined as:
               | 0.5 * x ** 2                  if abs(x) < beta
    huber(x) = |
               | beta * (abs(x) - 0.5 * beta)  otherwise
    Smooth L1 loss is equal to huber(x) / beta. This leads to the following
    differences:
     - As beta -> 0, Smooth L1 loss converges to L1 loss, while Huber loss
       converges to a constant 0 loss.
     - As beta -> +inf, Smooth L1 converges to a constant 0 loss, while Huber loss
       converges to L2 loss.
     - For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant
       slope of 1. For Huber loss, the slope of the L1 segment is beta.
    Smooth L1 loss can be seen as exactly L1 loss, but with the abs(x) < beta
    portion replaced with a quadratic function such that at abs(x) = beta, its
    slope is 1. The quadratic segment smooths the L1 loss near x = 0.

    Args:
        inp (Tensor): input tensor of any shape
        target (Tensor): target value tensor with the same shape as input
        beta (float): L1 to L2 change point.
            For beta values < 1e-5, L1 loss is computed.
        reduction (str): 'none' | 'mean' | 'sum'
             'none': No reduction will be applied to the output.
             'mean': The output will be averaged.
             'sum': The output will be summed.

    Returns:
        Tensor: The loss with the reduction option applied.

    Note:
        PyTorch's builtin "Smooth L1 loss" implementation does not actually
        implement Smooth L1 loss, nor does it implement Huber loss. It implements
        the special case of both in which they are equal (beta=1).
        See: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss.
     """
    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = torch.abs(inp - target)
    else:
        n = torch.abs(inp - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    return loss


class GIoULoss(torch.nn.Module):
    def __init__(self,
                 reduction: Optional[str] = None,
                 eps: float = 1e-7,
                 loss_weight: float = 1.,
                 ):
        """
        Generalized IoU Loss
        `Generalized Intersection over Union: A Metric and A Loss for Bounding
        Box Regression` https://arxiv.org/abs/1902.09630

        Args:
            eps: small constant for numerical stability

        Notes:
            Original paper uses lambda=10 to balance regression and cls losses
            for PASCAL VOC and COCO (not tuned for coco)

            `End-to-End Object Detection with Transformers` https://arxiv.org/abs/2005.12872
            "Our enhanced Faster-RCNN+ baselines use GIoU [38] loss along with
            the standard l1 loss for bounding box regression. We performed a grid search
            to find the best weights for the losses and the final models use only GIoU loss
            with weights 20 and 1 for box and proposal regression tasks respectively"
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute generalized iou loss

        Args:
            pred_boxes: predicted boxes (x1, y1, x2, y2, (z1, z2)) [N, dim * 2]
            target_boxes: target boxes (x1, y1, x2, y2, (z1, z2)) [N, dim * 2]

        Returns:
            Tensor: loss
        """
        loss = reduction_helper(
            torch.diag(generalized_box_iou(pred_boxes, target_boxes, eps=self.eps),
                       diagonal=0),
            reduction=self.reduction)
        return self.loss_weight * -1 * loss
