# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from detrex licensed under
# SPDX-FileCopyrightText: 2022, The IDEA Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List, Optional, Type, Union

import torch
import torch.nn as nn

from nndet.nn.layers.conv.base import BaseConvNormAct
from nndet.utils.typing import ND_INT


class ChannelMapper(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: List[int],
        in_features: List[int],
        out_channels: int,
        norm_layer: Optional[Union[Callable[..., Type[nn.Module]], str]] = None,
        activation: Optional[Union[Callable[..., Type[nn.Module]], str]] = None,
        kernel_size: ND_INT = 1,
        stride: ND_INT = 1,
        bias: bool = True,
        dilation: ND_INT = 1,
        groups: int = 1,
        num_outs: int = None,
        **kwargs,
    ):
        """
        Channel Mapper for reduce/increase channels of backbone features.
        This is used to reduce/increase the channels of backbone features.
        Args:
            dim: dimension, either 2 or 3
            input_shape: A dict which contains the backbone features meta
                infomation.
            in_features: A list contains the keys which maps the features output
                from the backbone
            out_channels: Number of output channels for each scale.
            kernel_size: Size of the convolving kernel for each scale.
            stride: Stride of convolution for each scale.
            bias: If True, adds a learnable bias to the output of each scale.
            groups: Number of blocked connections from input channels to output
                channels for each scale.
            dilation: Spacing between kernel elements for each scale.
            norm_layer: The norm layer used for each scale.
            activation: The activation layer used for each scale.
            num_outs: Number of output feature maps. There will be
                ``extra_convs`` when ``num_outs`` is larger than the length of
                ``in_features``.
        """
        super(ChannelMapper, self).__init__()
        self.extra_convs = None

        in_channels_per_feature = [in_channels[f] for f in in_features]

        if num_outs is None:
            num_outs = len(in_channels_per_feature)

        self.convs = nn.ModuleList()
        for in_channel in in_channels_per_feature:
            self.convs.append(
                BaseConvNormAct(
                    dim=dim,
                    in_channels=in_channel,
                    out_channels=out_channels,
                    norm=norm_layer,
                    act=activation,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    bias=bias,
                    dilation=dilation,
                    groups=groups,
                )
            )

        if num_outs > len(in_channels_per_feature):
            self.extra_convs = nn.ModuleList()
            for i in range(len(in_channels_per_feature), num_outs):
                if i == len(in_channels_per_feature):
                    in_channel = in_channels_per_feature[-1]
                else:
                    in_channel = out_channels
                self.extra_convs.append(
                    BaseConvNormAct(
                        dim=dim,
                        in_channels=in_channel,
                        out_channels=out_channels,
                        norm=norm_layer,
                        act=activation,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=(kernel_size - 1) // 2,
                        dilation=dilation,
                        groups=groups,
                        bias=bias,
                    )
                )

        self.input_shapes = in_channels
        self.in_features = in_features
        self.out_channels = out_channels
        assert len(self.convs) == len(self.in_features)

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward function for ChannelMapper
        Args:
            inputs (List[torch.Tensor]): The backbone feature maps.
        Return:
            tuple(torch.Tensor): A tuple of the processed features.
        """
        outs = [self.convs[i](inputs[in_feature]) for i, in_feature in enumerate(self.in_features)]
        if self.extra_convs:
            for i in range(len(self.extra_convs)):
                if i == 0:
                    outs.append(self.extra_convs[0](inputs[self.in_features[-1]]))
                else:
                    outs.append(self.extra_convs[i](outs[-1]))
        return outs


class GroupNormChannelMapper(ChannelMapper):
    """
    Channel Mapper that uses Group Norm
    """

    def __init__(
        self,
        dim: int,
        in_channels: List[int],
        in_features: List[int],
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = True,
        groups: int = 1,
        dilation: int = 1,
        activation: nn.Module = None,
        num_outs: int = None,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            in_channels=in_channels,
            in_features=in_features,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            groups=groups,
            dilation=dilation,
            norm_layer="Group",
            activation=activation,
            num_outs=num_outs,
        )
