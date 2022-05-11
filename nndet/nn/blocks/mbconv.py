from functools import reduce
from typing import Sequence

import torch

from nndet.nn.blocks.se import SELayer
from nndet.nn.conv import nd_pool


class MyFusedMBConv(torch.nn.Module):
    def __init__(
        self,
        conv,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        expansion: int = 2,
        reduction: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.fw = torch.nn.Sequential(
            conv(
                in_channels=in_channels,
                out_channels=in_channels * expansion,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                act_inplace=False,
                **kwargs,
            ),
            SELayer(
                dim=conv.dim,
                in_channels=in_channels * expansion,
                reduction=reduction,
            ),
            conv(
                in_channels=in_channels * expansion,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                act_inplace=False,
                **kwargs,
            ),
        )

        stride_prod = (
            reduce((lambda x, y: x * y), stride)
            if isinstance(stride, Sequence)
            else stride
        )
        if stride_prod > 1:
            self.shortcut = torch.nn.Sequential(
                nd_pool("Avg", dim=conv.dim, kernel_size=stride, stride=stride),
                conv(in_channels, out_channels, kernel_size=1, add_act=False),
            )
        else:
            self.shortcut = None

    def forward(self, x):
        res = x
        x = self.fw(x)

        if self.shortcut:
            res = self.shortcut(res)

        x += res
        return x
