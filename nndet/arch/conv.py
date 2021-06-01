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

import torch
import torch.nn as nn
from typing import Union, Callable, Any, Optional, Tuple, Sequence, Type

from nndet.arch.initializer import InitWeights_He
from nndet.arch.layers.norm import GroupNorm


NdParam = Union[int, Tuple[int, int], Tuple[int, int, int]]


class Generator:
    def __init__(self, conv_cls, dim: int):
        """
        Factory helper which saves the conv class and dimension to generate objects

        Args:
            conv_cls (callable): class of convolution
            dim (int): number of spatial dimensions (in general 2 or 3)
        """
        self.dim = dim
        self.conv_cls = conv_cls

    def __call__(self, *args, **kwargs) -> Any:
        """
        Create object

        Args:
            *args: passed to object
            **kwargs: passed to object

        Returns:
            Any
        """
        return self.conv_cls(self.dim, *args, **kwargs)


class BaseConvNormAct(torch.nn.Sequential):
    def __init__(self,
                 dim: int,
                 in_channels: int,
                 out_channels: int,
                 norm: Optional[Union[Callable[..., Type[nn.Module]], str]],
                 act: Optional[Union[Callable[..., Type[nn.Module]], str]],
                 kernel_size: Union[int, tuple],
                 stride: Union[int, tuple] = 1,
                 padding: Union[int, tuple] = 0,
                 dilation: Union[int, tuple] = 1,
                 groups: int = 1,
                 bias: bool = None,
                 transposed: bool = False,
                 norm_kwargs: Optional[dict] = None,
                 act_inplace: Optional[bool] = None,
                 act_kwargs: Optional[dict] = None,
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
                If None, the bias will be determined dynamicaly: False
                if a normalization follows otherwise True
            transposed: whether the convolution should be transposed or not
            norm_kwargs: keyword arguments for normalization layer
            act_inplace: whether to perform activation inplce or not
                If None, inplace will be determined dynamicaly: True
                if a normalization follows otherwise False
            act_kwargs: keyword arguments for non linearity layer.
            initializer: initilize weights
        """
        super().__init__()
        # process optional arguments
        norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        act_kwargs = {} if act_kwargs is None else act_kwargs

        if "inplace" in act_kwargs:
            raise ValueError("Use keyword argument to en-/disable inplace activations")
        if act_inplace is None:
            act_inplace = bool(norm is not None)
        act_kwargs["inplace"] = act_inplace

        # process dynamic values
        bias = bool(norm is None) if bias is None else bias

        conv = nd_conv(dim=dim,
                       in_channels=in_channels,
                       out_channels=out_channels,
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=padding,
                       dilation=dilation,
                       groups=groups,
                       bias=bias,
                       transposed=transposed
                       )
        self.add_module("conv", conv)

        if norm is not None:
            if isinstance(norm, str):
                _norm = nd_norm(norm, dim, out_channels, **norm_kwargs)
            else:
                _norm = norm(dim, out_channels, **norm_kwargs)
            self.add_module("norm", _norm)

        if act is not None:
            if isinstance(act, str):
                _act = nd_act(act, dim, **act_kwargs)
            else:
                _act = act(**act_kwargs)
            self.add_module("act", _act)

        if initializer is not None:
            self.apply(initializer)


class ConvInstanceRelu(BaseConvNormAct):
    def __init__(self,
                 dim: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, tuple],
                 stride: Union[int, tuple] = 1,
                 padding: Union[int, tuple] = 0,
                 dilation: Union[int, tuple] = 1,
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


class ConvGroupRelu(BaseConvNormAct):
    def __init__(self,
                 dim: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, tuple],
                 stride: Union[int, tuple] = 1,
                 padding: Union[int, tuple] = 0,
                 dilation: Union[int, tuple] = 1,
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
                "channels_per_group": norm_channels_per_group,
            },
            act_inplace=act_inplace,
            initializer=initializer,
        )


def nd_conv(dim: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, tuple],
            stride: Union[int, tuple] = 1,
            padding: Union[int, tuple] = 0,
            dilation: Union[int, tuple] = 1,
            groups: int = 1,
            bias: bool = True,
            transposed: bool = False,
            **kwargs,
            ) -> torch.nn.Module:
    """
    Convolution Wrapper to Switch accross dimensions and transposed by a
    single argument

    Args
        n_dim (int): number of dimensions the convolution should be chosen for
        in_channels (int): input channels
        out_channels (int): output_channels
        kernel_size (int or Iterable): size of convolution kernel
        stride (int or Iterable): convolution stride
        padding (int or Iterable): padding value
            (if input or output padding depends on whether the convolution
            is transposed or not)
        dilation (int or Iterable): convolution dilation
        groups (int): number of convolution groups
        bias (bool): whether to include bias or not
        transposed (bool): whether the convolution should be transposed or not

    Returns:
        torch.nn.Module: generated module

    See Also
        Torch Convolutions:
            * :class:`torch.nn.Conv1d`
            * :class:`torch.nn.Conv2d`
            * :class:`torch.nn.Conv3d`
            * :class:`torch.nn.ConvTranspose1d`
            * :class:`torch.nn.ConvTranspose2d`
            * :class:`torch.nn.ConvTranspose3d`
    """
    if transposed:
        transposed_str = "Transpose"
    else:
        transposed_str = ""

    conv_cls = getattr(torch.nn, f"Conv{transposed_str}{dim}d")

    return conv_cls(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding,
                    dilation=dilation, groups=groups, bias=bias, **kwargs)


def nd_pool(pooling_type: str, dim: int, *args, **kwargs) -> torch.nn.Module:
    """
    Wrapper to switch between different pooling types and convolutions by a single argument

    Args
        pooling_type (str): Type of Pooling, case sensitive.
                Supported values are
                * ``Max``
                * ``Avg``
                * ``AdaptiveAvg``
                * ``AdaptiveMax``
        n_dim (int): number of dimensions
        *args : positional arguments of the chosen pooling class
        **kwargs : keyword arguments of the chosen pooling class

    Returns:
        torch.nn.Module: generated module

    See Also
        Torch Pooling Classes:
            * :class:`torch.nn.MaxPool1d`
            * :class:`torch.nn.MaxPool2d`
            * :class:`torch.nn.MaxPool3d`
            * :class:`torch.nn.AvgPool1d`
            * :class:`torch.nn.AvgPool2d`
            * :class:`torch.nn.AvgPool3d`
            * :class:`torch.nn.AdaptiveMaxPool1d`
            * :class:`torch.nn.AdaptiveMaxPool2d`
            * :class:`torch.nn.AdaptiveMaxPool3d`
            * :class:`torch.nn.AdaptiveAvgPool1d`
            * :class:`torch.nn.AdaptiveAvgPool2d`
            * :class:`torch.nn.AdaptiveAvgPool3d`
    """
    pool_cls = getattr(torch.nn, f"{pooling_type}Pool{dim}d")
    return pool_cls(*args, **kwargs)


def nd_norm(norm_type: str, dim: int, *args, **kwargs) -> torch.nn.Module:
    """
    Wrapper to switch between different types of normalization and
    dimensions by a single argument

    Args
        norm_type (str): type of normalization, case sensitive.
            Supported types are:
                * ``Batch``
                * ``Instance``
                * ``LocalResponse``
                * ``Group``
                * ``Layer``
        n_dim (int, None): dimension of normalization input; can be None if normalization
            is dimension-agnostic (e.g. LayerNorm)
        *args : positional arguments of chosen normalization class
        **kwargs : keyword arguments of chosen normalization class

    Returns
        torch.nn.Module: generated module

    See Also
        Torch Normalizations:
                * :class:`torch.nn.BatchNorm1d`
                * :class:`torch.nn.BatchNorm2d`
                * :class:`torch.nn.BatchNorm3d`
                * :class:`torch.nn.InstanceNorm1d`
                * :class:`torch.nn.InstanceNorm2d`
                * :class:`torch.nn.InstanceNorm3d`
                * :class:`torch.nn.LocalResponseNorm`
                * :class:`nndet.arch.layers.norm.GroupNorm`
    """
    if dim is None:
        dim_str = ""
    else:
        dim_str = str(dim)

    if norm_type.lower() == "group":
        norm_cls = GroupNorm
    else:
        norm_cls = getattr(torch.nn, f"{norm_type}Norm{dim_str}d")
    return norm_cls(*args, **kwargs)


def nd_act(act_type: str, dim: int, *args, **kwargs) -> torch.nn.Module:
    """
    Helper to search for activations by string
    The dim parameter is ignored.
    Searches in torch.nn for activatio.

    Args:
        act_type: name of activation layer to look up.
        dim: ignored

    Returns:
        torch.nn.Module: activation module
    """
    act_cls = getattr(torch.nn, f"{act_type}")
    return act_cls(*args, **kwargs)


def nd_dropout(dim: int, p: float = 0.5, inplace: bool = False, **kwargs) -> torch.nn.Module:
    """
    Generate 1,2,3 dimensional dropout

    Args:
        dim (int): number of dimensions
        p (float): doupout probability
        inplace (bool): apply operation inplace
        **kwargs: passed to dropout

    Returns:
        torch.nn.Module: generated module
    """
    dropout_cls = getattr(torch.nn, "Dropout%dd" % dim)
    return dropout_cls(p=p, inplace=inplace, **kwargs)


def compute_padding_for_kernel(kernel_size: Union[int, Sequence[int]]) -> \
        Union[int, Tuple[int, int], Tuple[int, int, int]]:
    """
    Compute padding such that feature maps keep their size with stride 1

    Args:
        kernel_size: kernel size to compute padding for

    Returns:
        Union[int, Tuple[int, int], Tuple[int, int, int]]: computed padding
    """
    if isinstance(kernel_size, Sequence):
        padding = tuple([(i - 1) // 2 for i in kernel_size])
    else:
        padding = (kernel_size - 1) // 2
    return padding


def conv_kwargs_helper(norm: bool, activation: bool):
    """
    Helper to force disable normalization and activation in layers
    which have those by default

    Args:
        norm: en-/disable normalization layer
        activation: en-/disable activation layer

    Returns:
        dict: keyword arguments to pass to conv generator
    """
    kwargs = {
        "add_norm": norm,
        "add_act": activation,
    }
    return kwargs
