# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional

import torch.nn as nn

from nndet.nn.layers.conv.base import BaseConvNormAct
from nndet.utils.typing import ND_INT


class ConvBatchLReLU(BaseConvNormAct):
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
        add_norm: bool = True,
        add_act: bool = True,
        act_negative_slope: float = 1e-2,
        act_inplace: Optional[bool] = None,
        norm_eps: float = 1e-5,
        norm_affine: bool = True,
        norm_momentum: float = 0.1,
        initializer: Callable[[nn.Module], None] = None,
    ):
        """
        Baseclass for default ordering:
        conv -> norm -> activation

        Args
            dim: number of dimensions the convolution should be chosen for
            in_channels: input channels
            out_channels: output_channels
            norm: type of normalization. If None, no normalization will be applied
            kernel_size: size of convolution kernel
            act: class of non linearity; if None no actication is used.
            stride: convolution stride
            padding: padding value
                (if input or output padding depends on whether the convolution
                is transposed or not)
            dilation: convolution dilation
            groups: number of convolution groups
            bias: whether to include bias or not
                If None the bias will be determined dynamicaly: False
                if a normalization follows otherwise True
            transposed: whether the convolution should be transposed or not
            add_norm: add normalisation layer to conv block
            add_act: add activation layer to conv block
            act_inplace: whether to perform activation inplce or not
                If None, inplace will be determined dynamicaly: True
                if a normalization follows otherwise False
            norm_eps: instance norm eps (see pytorch for more info)
            norm_affine: instance affine parameter (see pytorch for more info)
            norm_momentum: momentum term of batch norm
            initializer: initilize weights
        """
        norm = "Batch" if add_norm else None
        act = "LeakyReLU" if add_act else None

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
            act_kwargs={"negative_slope": act_negative_slope},
            norm_kwargs={
                "eps": norm_eps,
                "affine": norm_affine,
                "momentum": norm_momentum,
            },
            act_inplace=act_inplace,
            initializer=initializer,
        )
