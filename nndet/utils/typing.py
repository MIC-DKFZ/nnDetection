# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional, Tuple, Type, Union

import torch

ND_INT = Union[int, Tuple[int, int], Tuple[int, int, int]]
ND_TUPLE_INT = Union[Tuple[int, int], Tuple[int, int, int]]  # no plain int allowed


class CONVSEQ(torch.nn.Sequential):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        norm: Optional[Union[Callable[..., Type[torch.nn.Module]], str]],
        act: Optional[Union[Callable[..., Type[torch.nn.Module]], str]],
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
        initializer: Callable[[torch.nn.Module], None] = None,
    ):
        """
        Provide interface to generate a Conv Sequence, usual sequences are:
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
        ...


class CONVGEN:
    """
    Provides a simple wrapper around conv sequences of various combinations
    (conv -> act -> norm), pre-norm, different activations / normalisations etc.
    """

    dim: int

    def __call__(self, **kwargs) -> CONVSEQ:
        ...
