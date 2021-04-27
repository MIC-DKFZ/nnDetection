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

from abc import abstractmethod
from typing import Sequence, Callable, Union, Tuple

from nndet.arch.conv import NdParam
from nndet.arch.blocks.res import ResBasic


class AbstractBlock(nn.Module):
    def __init__(self, out_channels: int, **kwargs):
        """
        Basic building block of the encoder
        """
        super().__init__(**kwargs)
        self.out_channels = out_channels

    def get_output_channels(self) -> int:
        """
        Determine number of output channels of block

        Returns:
            int: number of output channels
        """
        return self.out_channels


class StackedBlock(AbstractBlock):
    expansion = 2

    def __init__(self,
                 conv: Callable[[], nn.Module],
                 in_channels: int,
                 conv_kernel: NdParam,
                 stride: NdParam = None,
                 out_channels: int = None,
                 max_out_channels: int = None,
                 num_blocks: int = 1,
                 **kwargs):
        """
        Plain stack of convolutions. Strides > 1 are applied at the beginning
        by a strided convolution and the first convolution raises the number of
        channels to :param:`out_channels`.
        
        Args:
            conv: conv generator to use for internal convolutions
            in_channels: number of input channels
            conv_kernel: kernel size of convolution
            stride: Stride of first convolution. If None stride=1 will be used.
                Defaults to None.
            out_channels: If given, then number of output channels will be set 
                to this value. Otherwise the number of the input channels are 
                doubled. Defaults to None.
            max_out_channels: Maximum number of output channels.
                Defaults to None.
            num_blocks: Number of blocks. Defaults to 1.
        
        Raises:
            ValueError: raise if given output channels are larger than max
                output channels
        """
        super().__init__(out_channels=None) # out_channels will be overwritten later
        if (out_channels is not None and
            max_out_channels is not None and
            out_channels > max_out_channels):
            raise ValueError("Output channels can not be larger"
                             "than max output channels")
        if out_channels is None:
            out_channels = in_channels * self.expansion
        if max_out_channels is not None and out_channels > max_out_channels:
            out_channels = max_out_channels 
        if stride is None:
            stride = 1

        if not isinstance(conv_kernel, Sequence):
            conv_kernel = [conv_kernel] * conv.dim
        padding = tuple([(i - 1) // 2 for i in conv_kernel])

        _convs = []
        _convs.append(self.build_block(
            conv=conv, in_channels=in_channels, out_channels=out_channels,
            kernel_size=conv_kernel, stride=stride, padding=padding, **kwargs))
        for _ in range(num_blocks - 1):
            _convs.append(self.build_block(
                conv=conv, in_channels=out_channels, out_channels=out_channels,
                kernel_size=conv_kernel, stride=1, padding=padding, **kwargs))

        self.convs = nn.Sequential(*_convs)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward tensor
        
        Returns:
            torch.Tensor: output tensor
        """
        return self.convs(x)

    @abstractmethod
    def build_block(self, conv: Callable[[], nn.Module],
                    in_channels: int, out_channels: int,
                    kernel_size: NdParam,
                    stride: NdParam,
                    padding: NdParam,
                    ) -> nn.Module:
        raise NotImplementedError


class StackedConvBlock2(StackedBlock):
    def build_block(self, conv: Callable, in_channels: int,
                    out_channels: int, kernel_size: NdParam,
                    stride: NdParam, padding: NdParam,
                    **kwargs) -> nn.Module:
        """
        Build 2 consequtive convolutions

        Args:
            conv: generator for convolutions
            in_channels: number of input channels
            out_channels: number of output channels
            kernel_size: kernel size oh convolutions
            stride: stride of first convolution
            padding: padding of convolutions

        Returns:
            nn.Module: stacked convolutions
        """
        return torch.nn.Sequential(
            conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                 stride=stride, padding=padding, **kwargs),
            conv(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                 stride=1, padding=padding, **kwargs),
        )


class StackedConvBlock3(StackedBlock):
    def build_block(self, conv: Callable, in_channels: int,
                    out_channels: int, kernel_size: NdParam,
                    stride: NdParam, padding: NdParam,
                    **kwargs) -> nn.Module:
        """
        Build 2 consequtive convolutions

        Args:
            conv: generator for convolutions
            in_channels: number of input channels
            out_channels: number of output channels
            kernel_size: kernel size oh convolutions
            stride: stride of first convolution
            padding: padding of convolutions

        Returns:
            nn.Module: stacked convolutions
        """
        return torch.nn.Sequential(
            conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                 stride=stride, padding=padding, **kwargs),
            conv(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                 stride=1, padding=padding, **kwargs),
            conv(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                 stride=1, padding=padding, **kwargs),
        )


class StackedResidualBlock(StackedBlock):
    def build_block(self, conv: Callable[[], nn.Module], in_channels: int,
                    out_channels: int, kernel_size: NdParam,
                    stride: NdParam, padding: NdParam,
                    **kwargs) -> nn.Module:
        """
        Build Residual Block

        Args:
            conv: generator for convolutions
            in_channels: number of input channels
            out_channels: number of output channels
            kernel_size: kernel size oh convolutions
            stride: stride of first convolution
            padding: padding of convolutions

        Returns:
            nn.Module: stacked convolutions
        """
        return ResBasic(conv=conv, in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=padding, **kwargs)


class StackedConvBlock(AbstractBlock):
    expansion = 2

    def __init__(self,
                 conv: Callable[[], nn.Module],
                 in_channels: int,
                 conv_kernel: Union[Tuple[int], int],
                 stride: Union[Tuple[int], int] = None,
                 out_channels: int = None,
                 max_out_channels: int = None,
                 num_blocks: int = 2,
                 **kwargs):
        """
        Plain stack of convolutions. Strides > 1 are applied at the beginning
        by a strided convolution and the first convolution raises the number of
        channels to :param:`out_channels`.
        
        Args:
            conv: conv generator to use for internal convolutions
            in_channels: number of input channels
            conv_kernel: kernel size of convolution
            stride: Stride of first convolution. If None stride=1 will be used.
                Defaults to None.
            out_channels: If given, then number of output channels will be set 
                to this value. Otherwise the number of the input channels are 
                doubled. Defaults to None.
            max_out_channels: Maximum number of output channels.
                Defaults to None.
            num_blocks: Number of convolutions. Defaults to 2.
        
        Raises:
            ValueError: raise if given output channels are larger than max
                output channels
        """
        super().__init__(out_channels=None) # out_channels will be overwritten later
        if (out_channels is not None and
            max_out_channels is not None and
            out_channels > max_out_channels):
            raise ValueError("Output channels can not be larger"
                             "than max output channels")
        if out_channels is None:
            out_channels = in_channels * self.expansion
        if max_out_channels is not None and out_channels > max_out_channels:
            out_channels = max_out_channels 
        if stride is None:
            stride = 1

        if not isinstance(conv_kernel, Sequence):
            conv_kernel = [conv_kernel] * conv.dim
        padding = tuple([(i - 1) // 2 for i in conv_kernel])

        _convs = []
        _convs.append(conv(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=conv_kernel,
                           stride=stride,
                           padding=padding,
                           **kwargs))
        for _ in range(num_blocks - 1):
            _convs.append(conv(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=conv_kernel,
                               stride=1,
                               padding=padding,
                               **kwargs))

        self.convs = nn.Sequential(*_convs)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward tensor
        
        Returns:
            torch.Tensor: output tensor
        """
        return self.convs(x)
