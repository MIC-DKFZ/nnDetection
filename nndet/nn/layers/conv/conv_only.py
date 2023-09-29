# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import torch.nn as nn

from nndet.nn.layers.conv.base import BaseConvNormAct
from nndet.utils.typing import ND_INT


class ConvOnly(BaseConvNormAct):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: ND_INT,
        stride: ND_INT = 1,
        padding: ND_INT = 0,
        dilation: ND_INT = 1,
        groups: int = 1,
        bias: bool = None,
        transposed: bool = False,
        initializer: Callable[[nn.Module], None] = None,
    ):
        """
        Baseclass for default ordering:
        conv -> norm -> activation

        Args
            dim: number of dimensions the convolution should be chosen for
            in_channels: input channels
            out_channels: output_channels
            norm: type of normalization. If None, no normalization will be
                applied
            kernel_size: size of convolution kernel
            act: class of non linearity; if None no activation is used.
            stride: convolution stride
            padding: padding value (if input or output padding depends on
                whether the convolution is transposed or not)
            dilation: convolution dilation
            groups: number of convolution groups
            bias: whether to include bias or not
                If None the bias will be determined dynamically: False
                if a normalization follows otherwise True
            transposed: whether the convolution should be transposed or not
            initializer: initialize weights
        """
        norm = None
        act = None

        super().__init__(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            transposed=transposed,
            norm=norm,
            act=act,
            initializer=initializer,
        )
