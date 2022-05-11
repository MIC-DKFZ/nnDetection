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


class ConvInstanceRelu(BaseConvNormAct):
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
            initializer: initilize weights
        """
        norm = "Instance" if add_norm else None
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
            },
            act_inplace=act_inplace,
            initializer=initializer,
        )


class ConvInstanceSiLU(BaseConvNormAct):
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
            initializer: initilize weights
        """
        norm = "Instance" if add_norm else None
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
            },
            act_inplace=act_inplace,
            initializer=initializer,
        )


class ConvInstanceLReLU(BaseConvNormAct):
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
        act_negative_slope: float = 1e-2,
        norm_eps: float = 1e-5,
        norm_affine: bool = True,
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
            act_negative_slope: negative slope of leaky relu activation
            norm_eps: instance norm eps (see pytorch for more info)
            norm_affine: instance affine parameter (see pytorch for more info)
            initializer: initilize weights
        """
        norm = "Instance" if add_norm else None
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
            },
            act_inplace=act_inplace,
            initializer=initializer,
        )


class ConvInstanceSwish(BaseConvNormAct):
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
            initializer: initilize weights
        """
        norm = "Instance" if add_norm else None
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
            },
            act_inplace=act_inplace,
            initializer=initializer,
        )


class ConvInstanceMish(BaseConvNormAct):
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
            initializer: initilize weights
        """
        norm = "Instance" if add_norm else None
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
            },
            act_inplace=act_inplace,
            initializer=initializer,
        )
