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

"""
Don't use these. Next nnDetection Version will introduce better/fixed implementations.
"""
import torch
import torch.nn as nn

from typing import Sequence, Callable, Optional
from functools import reduce 
from loguru import logger

from nndet.arch.conv import nd_pool
from nndet.arch.conv import NdParam


class ResBasic(nn.Module):
    def __init__(self,
                 conv: Callable,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: NdParam,
                 stride: NdParam,
                 padding: NdParam,
                 attention: Optional[nn.Module] = None,
                 ):
        """
        Build a plan residual block
        Zero init norm according to https://arxiv.org/abs/1706.02677
        Avg pool in downsampling path https://arxiv.org/pdf/1812.01187.pdf

        Args:
            conv: generator for convolutions
            in_channels: number of input channels
            out_channels: number of output channels
            kernel_size: kernel size oh convolutions
            stride: stride of first convolution
            padding: padding of convolutions
            attention: additional attention layer applied after convolutions
        """
        super().__init__()
        logger.warning("ResidualBlock uses normal relu! This might not be "
                       "desired if conv uses a different non linearity")

        self.conv1 = conv(in_channels, out_channels, kernel_size=kernel_size,
                          padding=padding, stride=stride)
        self.conv2 = conv(out_channels, out_channels, kernel_size=kernel_size,
                          padding=padding, relu=None)
        self.relu = nn.ReLU(inplace=True)

        stride_prod = (reduce((lambda x, y: x * y), stride)
                       if isinstance(stride, Sequence) else stride)
        if stride_prod > 1:
            self.shortcut = nn.Sequential(
                nd_pool("Avg", dim=conv.dim, kernel_size=stride, stride=stride),
                conv(in_channels, out_channels, kernel_size=1, relu=None),
                )
        else:
            self.shortcut = None

        self.attention = attention
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward input

        Args:
            x (torch.Tensor) : input tensor

        Returns:
            torch.Tensor: output tensor
        """
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.attention:
            out = self.attention(out)
        if self.shortcut:
            residual = self.shortcut(x)

        out += residual
        out = self.relu(out)
        return out

    def init_weights(self) -> None:
        try:
            torch.nn.init.zeros_(self.conv2.norm.weight)
        except:
            logger.info(f"Zero init of last norm layer {self.conv2.norm} failed")


class ResBottleneck(nn.Module):
    def __init__(self,
                 conv: Callable,
                 in_channels: int,
                 internal_channels: int,
                 kernel_size: NdParam,
                 stride: NdParam,
                 padding: NdParam,
                 expansion: int = 1,
                 attention: Optional[nn.Module] = None,
                 ):
        """
        Build a bottleneck residual block
        Zero init norm according to https://arxiv.org/abs/1706.02677
        Avg pool in downsampling path https://arxiv.org/pdf/1812.01187.pdf

        in_channels -> internal_channels -> internal_channels * expansion

        Args:
            conv: generator for convolutions
            in_channels: number of input channels
            internal_channels: number of internal channels to use.
                The number of output channels will be
                internal_channels * expansion
            kernel_size: kernel size oh convolutions
            stride: stride of first convolution
            padding: padding of convolutions
            expansion: expansion for last conv block. Default expansion
                is one to be compatible with modular encoder! Original
                implementation uses expansion=4.
            attention: additional attention layer applied after convolutions
        """
        super().__init__()
        logger.warning("ResidualBlock uses normal relu! This might not be "
                       "desired if conv uses a different non linearity")

        out_channels = internal_channels * expansion
        self.conv1 = conv(in_channels, internal_channels,
                          kernel_size=1, padding=0, stride=1,
                          )
        self.conv2 = conv(internal_channels, internal_channels,
                          kernel_size=kernel_size, padding=padding, stride=stride,
                          )
        self.conv3 = conv(internal_channels, out_channels,
                          kernel_size=1, padding=0, relu=None, stride=1,
                          )
        self.relu = nn.ReLU(inplace=True)

        # downsampling path
        stride_prod = (reduce((lambda x, y: x * y), stride)
                       if isinstance(stride, Sequence) else stride)
        if stride_prod > 1:
            self.shortcut = nn.Sequential(
                nd_pool("Avg", dim=conv.dim, kernel_size=stride, stride=stride),
                conv(in_channels, out_channels, kernel_size=1, relu=None),
                )
        else:
            self.shortcut = None

        self.attention = attention
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward input

        Args:
            x (torch.Tensor) : input tensor

        Returns:
            torch.Tensor: output tensor
        """
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.attention:
            out = self.attention(out)
        if self.shortcut:
            residual = self.shortcut(x)

        out += residual
        out = self.relu(out)
        return out

    def init_weights(self) -> None:
        try:
            torch.nn.init.zeros_(self.conv2.norm.weight)
        except:
            logger.info(f"Zero init of last norm layer {self.conv2.norm} failed")
