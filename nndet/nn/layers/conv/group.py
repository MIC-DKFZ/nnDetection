# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional

import torch.nn as nn
from loguru import logger

from nndet.nn.layers.conv.base import BaseConvNormAct
from nndet.nn.ops.activation import Swish
from nndet.utils.typing import ND_INT

try:
    from torch.nn import Mish

    torch_mish = True
except ImportError:
    from nndet.nn.ops.activation import Mish

    torch_mish = False


class ConvGroupRelu(BaseConvNormAct):
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
        act_inplace: Optional[bool] = None,
        norm_eps: float = 1e-5,
        norm_affine: bool = True,
        num_groups: Optional[int] = None,
        norm_channels_per_group: Optional[int] = 16,
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
            norm_channels_per_group: channels per group for group norm.
                Default 16.
            initializer: initilize weights
        """
        if num_groups is not None and norm_channels_per_group != 16:
            raise ValueError("Can not use both `num_groups` and `channels_per_group`")
        if num_groups is not None and norm_channels_per_group == 16:
            norm_channels_per_group = None

        norm = "Group" if add_norm else None
        act = "ReLU" if add_act else None

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
            norm_kwargs={
                "eps": norm_eps,
                "affine": norm_affine,
                "num_groups": num_groups,
                "channels_per_group": norm_channels_per_group,
            },
            act_inplace=act_inplace,
            initializer=initializer,
        )


class ConvGroupLReLU(BaseConvNormAct):
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
        norm_channels_per_group: int = 16,
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
            act_negative_slope: negative slope for LRelu activation
            norm_eps: instance norm eps (see pytorch for more info)
            norm_affine: instance affine parameter (see pytorch for more info)
            norm_channels_per_group: channels per group for group norm
            initializer: initilize weights
        """
        norm = "Group" if add_norm else None
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
                "channels_per_group": norm_channels_per_group,
            },
            act_inplace=act_inplace,
            initializer=initializer,
        )


class ConvGroupSiLU(BaseConvNormAct):
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
        act_inplace: Optional[bool] = None,
        norm_eps: float = 1e-5,
        norm_affine: bool = True,
        norm_channels_per_group: int = 16,
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
            norm_channels_per_group: channels per group for group norm
            initializer: initilize weights
        """
        norm = "Group" if add_norm else None
        act = "SiLU" if add_act else None

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
            norm_kwargs={
                "eps": norm_eps,
                "affine": norm_affine,
                "channels_per_group": norm_channels_per_group,
            },
            act_inplace=act_inplace,
            initializer=initializer,
        )


class ConvGroupSwish(BaseConvNormAct):
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
        act_inplace: Optional[bool] = None,
        norm_eps: float = 1e-5,
        norm_affine: bool = True,
        norm_channels_per_group: int = 16,
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
            norm_channels_per_group: channels per group for group norm
            initializer: initilize weights
        """
        norm = "Group" if add_norm else None
        act = Swish if add_act else None

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
            norm_kwargs={
                "eps": norm_eps,
                "affine": norm_affine,
                "channels_per_group": norm_channels_per_group,
            },
            act_inplace=act_inplace,
            initializer=initializer,
        )


class ConvGroupMish(BaseConvNormAct):
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
        act_inplace: Optional[bool] = None,
        norm_eps: float = 1e-5,
        norm_affine: bool = True,
        norm_channels_per_group: int = 16,
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
            norm_channels_per_group: channels per group for group norm
            initializer: initilize weights
        """
        norm = "Group" if add_norm else None
        act = Mish if add_act else None
        if not torch_mish:
            logger.error(
                "Could not import Mish from Torch, update to PyTorch 1.9 or later!"
                "The current implementation uses too much memory."
            )

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
            norm_kwargs={
                "eps": norm_eps,
                "affine": norm_affine,
                "channels_per_group": norm_channels_per_group,
            },
            act_inplace=act_inplace,
            initializer=initializer,
        )
