from typing import Sequence

import torch

from nndet.nn.ops.norm import GroupNorm
from nndet.utils.enums import InterpolationMode
from nndet.utils.typing import ND_INT


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

    def __call__(self, *args, **kwargs) -> torch.nn.Module:
        """
        Create object

        Args:
            *args: passed to object
            **kwargs: passed to object

        Returns:
            Any
        """
        return self.conv_cls(self.dim, *args, **kwargs)


def nd_conv(
    dim: int,
    in_channels: int,
    out_channels: int,
    kernel_size: ND_INT,
    stride: ND_INT = 1,
    padding: ND_INT = 0,
    dilation: ND_INT = 1,
    groups: int = 1,
    bias: bool = True,
    transposed: bool = False,
    **kwargs,
) -> torch.nn.Module:
    """
    Convolution Wrapper to Switch accross dimensions and transposed by a
    single argument

    Args
        dim: number of dimensions the convolution should be chosen for
        in_channels: input channels
        out_channels: output_channels
        kernel_size: size of convolution kernel
        stride: convolution stride
        padding: padding value
            (if input or output padding depends on whether the convolution
            is transposed or not)
        dilation: convolution dilation
        groups: number of convolution groups
        bias: whether to include bias or not
        transposed: whether the convolution should be transposed or not

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

    return conv_cls(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        **kwargs,
    )


def nd_pool(
    pooling_type: str,
    dim: int,
    *args,
    **kwargs,
) -> torch.nn.Module:
    """
    Wrapper to switch between different pooling types and convolutions by a single argument

    Args
        pooling_type: Type of Pooling, case sensitive.
                Supported values are
                * ``Max``
                * ``Avg``
                * ``AdaptiveAvg``
                * ``AdaptiveMax``
        n_dim: number of dimensions
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


def nd_norm(
    norm_type: str,
    dim: int,
    *args,
    **kwargs,
) -> torch.nn.Module:
    """
    Wrapper to switch between different types of normalization and
    dimensions by a single argument

    Args
        norm_type: type of normalization, case sensitive.
            Supported types are:
                * ``Batch``
                * ``Instance``
                * ``LocalResponse``
                * ``Group``
                * ``Layer``
        dim: dimension of normalization input; can be None if normalization
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


def nd_act(
    act_type: str,
    dim: int,
    *args,
    **kwargs,
) -> torch.nn.Module:
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


def nd_dropout(
    dim: int,
    p: float = 0.5,
    inplace: bool = False,
    **kwargs,
) -> torch.nn.Module:
    """
    Generate 1,2,3 dimensional dropout

    Args:
        dim: number of dimensions
        p: doupout probability
        inplace: apply operation inplace
        **kwargs: passed to dropout

    Returns:
        torch.nn.Module: generated module
    """
    dropout_cls = getattr(torch.nn, "Dropout%dd" % dim)
    return dropout_cls(p=p, inplace=inplace, **kwargs)


def compute_padding_for_kernel(
    kernel_size: ND_INT,
) -> ND_INT:
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


def torch_interpolation(mode: InterpolationMode, dim: int) -> str:
    """
    Map enum interpolation modes to torch string interpolation modes

    Args:
        mode: desired interpolation mode
        dim: number of spatial dimensions

    Raises:
        ValueError: Unsupported mode

    Returns:
        str: string for interpolation mode
    """
    if mode == InterpolationMode.NEAREST:
        return "nearest"
    elif mode == InterpolationMode.LINEAR:
        return "bilinear" if dim == 2 else "trilinear"
    elif mode == InterpolationMode.CUBIC:
        return "bicubic" if dim == 2 else "tricubic"
    else:
        raise ValueError(f"Interpolation mode {mode} not compatible with torch.")
