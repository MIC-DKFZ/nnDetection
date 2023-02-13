# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional

import pytest
import torch
from loguru import logger
from torch import Tensor
from torch.cuda.amp import autocast

from nndet.losses.mask.functional.dice import soft_dice
from nndet.losses.ops import Loss, reduction_helper
from nndet.losses.segmentation.dice import BDiceSegLoss, DiceSegLoss

###
# This DiceLoss computation is a slightly adapted version from nnU-Net V1
# and was used in nnDetection V1
# We use this to compare directly against it here
###

###############################################################################
# Previous Implementation
################################################################################


class SoftDiceSegLoss(Loss):
    def __init__(
        self,
        nonlin: Callable = None,
        batch_dice: bool = False,
        do_bg: bool = False,
        smooth_nom: float = 1e-5,
        smooth_denom: float = 1e-5,
        loss_weight: float = 1.0,
        loss_fp32: bool = True,
        reduction: str = "mean",
    ):
        """
        Soft dice loss

        Args:
            nonlin: treat batch as pseudo volume. Defaults to False.
            do_bg: include background for dice computation. Defaults to True.
            smooth_nom: smoothing for nominator
            smooth_denom: smoothing for denominator
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            reduction: 'mean' | 'sum'
                - 'mean': The output will be averaged.
                - 'sum': The output will be summed.
                - 'none': NOT supported
        """
        super().__init__(
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
            reduction=reduction,
        )

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.nonlin = nonlin
        self.smooth_nom = smooth_nom
        self.smooth_denom = smooth_denom
        logger.info(f"Running batch dice {self.batch_dice} and " f"do bg {self.do_bg} in dice loss.")
        if self.reduction.lower() == "none":
            raise ValueError(f"SoftDice does not support reduction {reduction}.")

    def forward(
        self,
        inp: torch.Tensor,
        target: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ):
        """
        Compute loss

        Args:
            inp: predictions [N, C, dims]
            target: ground truth [N, dims]
            loss_mask: binary mask. Defaults to None.

        Returns:
            torch.Tensor: soft dice loss
        """
        if self.loss_fp32:
            with autocast(enabled=False):
                loss = self.loss_weight * soft_dice(
                    inp.float(),
                    target.float(),
                    do_bg=self.do_bg,
                    batch_dice=self.batch_dice,
                    nonlin=self.nonlin,
                    smooth_nom=self.smooth_nom,
                    smooth_denom=self.smooth_denom,
                    reduction=self.reduction,
                    loss_mask=loss_mask,
                )
        else:
            loss = self.loss_weight * soft_dice(
                inp,
                target,
                do_bg=self.do_bg,
                batch_dice=self.batch_dice,
                nonlin=self.nonlin,
                smooth_nom=self.smooth_nom,
                smooth_denom=self.smooth_denom,
                reduction=self.reduction,
                loss_mask=loss_mask,
            )
        return loss


def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z)))
    or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z))

        - `net_output`
        - `gt`
        - `axes`
        - `mask` : mask must be 1 for valid pixels and 0 for invalid pixels
        - `square` : if True then fp, tp and fn will be squared before summation
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp**2
        fp = fp**2
        fn = fn**2

    tp = tp.sum(dim=axes, keepdim=False)
    fp = fp.sum(dim=axes, keepdim=False)
    fn = fn.sum(dim=axes, keepdim=False)
    return tp, fp, fn


def soft_dice(
    inp: Tensor,
    target: Tensor,
    do_bg: bool = False,
    batch_dice: bool = False,
    nonlin: Optional[Callable] = None,
    smooth_nom: float = 1e-5,
    smooth_denom: float = 1e-5,
    loss_mask: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Tensor:
    shp_x = inp.shape

    if batch_dice:
        axes = [0] + list(range(2, len(shp_x)))
    else:
        axes = list(range(2, len(shp_x)))

    if nonlin is not None:
        inp = nonlin(inp)

    tp, fp, fn = get_tp_fp_fn(inp, target, axes, loss_mask, False)

    nominator = 2 * tp + smooth_nom
    denominator = 2 * tp + fp + fn + smooth_denom

    dc = nominator / denominator

    if not do_bg:
        if batch_dice:
            dc = dc[1:]
        else:
            dc = dc[:, 1:]
    return reduction_helper(dc, reduction=reduction) * (-1)


###############################################################################
# Tests
################################################################################


@pytest.mark.parametrize("batch_dice", [True, False])
@pytest.mark.parametrize("smooth_nom", [0.0, 1e-5])
@pytest.mark.parametrize("smooth_denom", [0.0, 1e-5])
@pytest.mark.parametrize("do_bg", [True, False])
def test_prev_softmax(
    batch_dice: bool,
    smooth_nom: float,
    smooth_denom: float,
    do_bg: bool,
):
    prev_dice_loss = SoftDiceSegLoss(
        nonlin=torch.nn.Softmax(dim=1),
        batch_dice=batch_dice,
        smooth_nom=smooth_nom,
        smooth_denom=smooth_denom,
        do_bg=do_bg,
        reduction="mean",
    )
    dice_loss = DiceSegLoss(
        batch_dice=batch_dice,
        smooth_nom=smooth_nom,
        smooth_denom=smooth_denom,
        do_bg=do_bg,
        reduction="mean",
    )

    for i in range(100):
        torch.manual_seed(i)
        preds = torch.rand(4, 3, 8, 8, 8)
        targets = torch.randint(low=0, high=3, size=(4, 8, 8, 8))
        prev_loss = prev_dice_loss(preds.clone(), targets.clone())
        loss = dice_loss(preds, targets)
        assert torch.allclose(prev_loss, loss)


@pytest.mark.parametrize("batch_dice", [True, False])
@pytest.mark.parametrize("do_bg", [True, False])
@pytest.mark.parametrize("smooth_nom", [0.0, 1e-5])
@pytest.mark.parametrize("smooth_denom", [0.0, 1e-5])
def test_prev_sigmoid(
    batch_dice: bool,
    smooth_nom: float,
    smooth_denom: float,
    do_bg: bool,
):
    prev_dice_loss = SoftDiceSegLoss(
        nonlin=torch.nn.Sigmoid(),
        batch_dice=batch_dice,
        do_bg=do_bg,
        smooth_nom=smooth_nom,
        smooth_denom=smooth_denom,
        reduction="mean",
    )
    dice_loss = BDiceSegLoss(
        batch_dice=batch_dice,
        do_bg=do_bg,
        smooth_nom=smooth_nom,
        smooth_denom=smooth_denom,
        reduction="mean",
    )

    for i in range(100):
        torch.manual_seed(i)
        preds = torch.rand(4, 3, 8, 8, 8)
        targets = torch.randint(low=0, high=3, size=(4, 8, 8, 8))
        prev_loss = prev_dice_loss(preds.clone(), targets.clone())
        loss = dice_loss(preds, targets)
        assert torch.allclose(prev_loss, loss)
