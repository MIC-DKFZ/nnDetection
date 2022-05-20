from typing import Callable, Optional, Type, Union

import torch
import torch.nn as nn

from nndet.nn.layers.wrapper import nd_act, nd_conv, nd_norm
from nndet.utils.typing import ND_INT


class BaseConvNormAct(torch.nn.Sequential):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        norm: Optional[Union[Callable[..., Type[nn.Module]], str]],
        act: Optional[Union[Callable[..., Type[nn.Module]], str]],
        kernel_size: ND_INT,
        stride: ND_INT = 1,
        padding: ND_INT = 0,
        dilation: ND_INT = 1,
        groups: int = 1,
        bias: bool = None,
        transposed: bool = False,
        norm_kwargs: Optional[dict] = None,
        act_inplace: Optional[bool] = None,
        act_kwargs: Optional[dict] = None,
        initializer: Callable[[nn.Module], None] = None,
    ):
        """
        Baseclass for default ordering: conv -> norm -> activation

        Args
            dim: number of dimensions the convolution should be chosen for
            in_channels: input channels
            out_channels: output_channels
            norm: type of normalization. If None, no normalization will be
                applied
            kernel_size: size of convolution kernel
            act: class of non linearity; if None no actication is used.
            stride: convolution stride
            padding: padding value (if input or output padding depends on
                whether the convolution is transposed or not)
            dilation: convolution dilation
            groups: number of convolution groups
            bias: whether to include bias or not If None, the bias will be
                determined dynamicaly: False; if a normalization follows
                otherwise True;
            transposed: whether the convolution should be transposed or not
            norm_kwargs: keyword arguments for normalization layer
            act_inplace: whether to perform activation inplce or not; If None,
                inplace will be determined dynamicaly: True; if a normalization
                follows otherwise False;
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

        conv = nd_conv(
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
