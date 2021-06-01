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

from loguru import logger
import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable


def one_hot_smooth_batch(data, num_classes: int, smoothing: float = 0.0):
    shape = data.shape
    targets = torch.empty(size=(shape[0], num_classes, *shape[1:]), device=data.device)\
        .fill_(smoothing / num_classes)\
        .scatter_(1, data.long().unsqueeze(1), 1. - smoothing)
    return targets


def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
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
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = tp.sum(dim=axes, keepdim=False)
    fp = fp.sum(dim=axes, keepdim=False)
    fn = fn.sum(dim=axes, keepdim=False)
    return tp, fp, fn


class SoftDiceLoss(nn.Module):
    def __init__(self,
                 nonlin: Callable = None,
                 batch_dice: bool = False, 
                 do_bg: bool = False,
                 smooth_nom: float = 1e-5,
                 smooth_denom: float = 1e-5,
                 ):
        """
        Soft dice loss
        
        Args:
            nonlin: treat batch as pseudo volume. Defaults to False.
            do_bg: include background for dice computation. Defaults to True.
            smooth_nom: smoothing for nominator
            smooth_denom: smoothing for denominator
        """
        super().__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.nonlin = nonlin
        self.smooth_nom = smooth_nom
        self.smooth_denom = smooth_denom
        logger.info(f"Running batch dice {self.batch_dice} and "
                    f"do bg {self.do_bg} in dice loss.")

    def forward(self,
                inp: torch.Tensor,
                target: torch.Tensor,
                loss_mask: torch.Tensor=None,
                ):
        """
        Compute loss
        
        Args:
            inp (torch.Tensor): predictions
            target (torch.Tensor): ground truth
            loss_mask ([torch.Tensor], optional): binary mask. Defaults to None.
        
        Returns:
            torch.Tensor: soft dice loss
        """
        shp_x = inp.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.nonlin is not None:
            inp = self.nonlin(inp)

        tp, fp, fn = get_tp_fp_fn(inp, target, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth_nom
        denominator = 2 * tp + fp + fn + self.smooth_denom

        dc = nominator / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return 1 - dc


class TopKLoss(torch.nn.CrossEntropyLoss):
    def __init__(self,
                 topk: float,
                 loss_weight: float = 1.,
                 **kwargs,
                 ):
        """
        Uses topk percent of values to compute CE loss
        (expects pre softmax logits!)

        Args:
            topk: percentage of all entries to use for loss computation
            loss_weight: scalar to balance multiple losses
        """
        if "reduction" in kwargs:
            raise ValueError("Reduction is not supported in TopKLoss."
                             "This will always return the mean!")
        super().__init__(
            reduction="none",
            **kwargs,
        )
        if topk < 0 or topk > 1:
            raise ValueError("topk needs to be in the range [0, 1].")
        self.topk = topk
        self.loss_weight = loss_weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Compute CE loss and uses mean of topk percent of the entries

        Args:
            input: logits for all foreground classes [N, C, *]
            target: target classes. 0 is treated as background, >0 are
                treated as foreground classes. [N, *]

        Returns:
            Tensor: final loss
        """
        losses = super().forward(input, target)

        k = int(losses.numel() * self.topk)
        return self.loss_weight * losses.view(-1).topk(k=k, sorted=False)[0].mean()


class TopKLossSigmoid(torch.nn.BCEWithLogitsLoss):
    def __init__(self,
                 num_classes: int,
                 topk: float,
                 smoothing: float = 0.0,
                 loss_weight: float = 1.,
                 **kwargs,
                 ):
        """
        Uses topk percent of values to compute BCE loss with one hot
        (support multi class through one hot, expects pre sigmoid logits!)

        Args:
            num_classes: number of classes
            topk: percentage of all entries to use for loss computation
            smoothing:  label smoothing
            loss_weight: scalar to balance multiple losses
        """
        if "reduction" in kwargs:
            raise ValueError("Reduction is not supported in TopKLoss."
                             "This will always return the mean!")
        super().__init__(
            reduction="none",
            **kwargs,
        )
        self.smoothing = smoothing
        if smoothing > 0:
            logger.info(f"Running label smoothing with smoothing: {smoothing}")
        self.num_classes = num_classes

        self.topk = topk
        self.loss_weight = loss_weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Compute BCE loss based on one hot encoding of foreground(!) classes
        and uses mean of topk percent of the entries

        Args:
            input: logits for all foreground(!) classes [N, C, *]
            target: target classes [N, *]. Targets will be encoded with one
                hot and 0 is treated as the background class and removed.

        Returns:
            Tensor: final loss
        """
        target_one_hot = one_hot_smooth_batch(
            target, num_classes=self.num_classes + 1, smoothing=self.smoothing)  # [N, C + 1]
        target_one_hot = target_one_hot[:, 1:]  # background is implicitly encoded
        losses = super().forward(input, target_one_hot.float())

        k = int(losses.numel() * self.topk)
        return self.loss_weight * losses.view(-1).topk(k=k, sorted=False)[0].mean()

