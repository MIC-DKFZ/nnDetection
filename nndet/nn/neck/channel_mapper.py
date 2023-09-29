# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from detrex licensed under
# SPDX-FileCopyrightText: 2022, The IDEA Authors
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

import torch
import torch.nn as nn

from nndet.nn.layers.wrapper import compute_padding_for_kernel
from nndet.utils.typing import CONVGEN, ND_INT


class ChannelMapper(nn.Module):
    def __init__(
        self,
        conv: CONVGEN,
        in_channels: List[int],
        num_in_features: int,
        kernel_size: ND_INT,
        out_channels: int,
        num_outs: Optional[int] = None,
        **kwargs,
    ):
        """
        Channel Mapper for reducing/increasing channels of backbone features to
        the same dimension.

        Args:
            conv: generator to create the convolution
            in_channels: list containing the number of channels of the backbone
                features
            num_in_features: number of backbone feature maps that should be used
            kernel_size: Size of the convolution kernel for each scale.
            out_channels: Number of output channels for each scale.
            num_outs: (Optional) Number of output feature maps. There will be
                ``extra_convs`` when ``num_outs`` is larger than the length of
                ``in_features``.
            **kwargs: kwargs used by the convolution generator, could be
                'stride', 'groups', 'bias' or other
        """
        super(ChannelMapper, self).__init__()
        self.extra_convs = None
        # get the last num_in_features
        in_features = [i for i in range(len(in_channels) - num_in_features, len(in_channels), 1)]
        in_channels_per_feature = [in_channels[f] for f in in_features]

        if num_outs is None:
            num_outs = len(in_channels_per_feature)

        self.convs = nn.ModuleList()
        padding = compute_padding_for_kernel(kernel_size)
        for in_channel in in_channels_per_feature:
            self.convs.append(
                conv(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    **kwargs,
                )
            )

        if num_outs > num_in_features:
            self.extra_convs = nn.ModuleList()
            for i in range(len(in_channels_per_feature), num_outs):
                if i == len(in_channels_per_feature):
                    in_channel = in_channels_per_feature[-1]
                else:
                    in_channel = out_channels
                self.extra_convs.append(
                    conv(
                        in_channels=in_channel,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        **kwargs,
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
            inputs: The backbone feature maps.

        Returns:
            tuple: A tuple of the processed features.
        """
        outs = [self.convs[i](inputs[in_feature]) for i, in_feature in enumerate(self.in_features)]
        if self.extra_convs:
            for i in range(len(self.extra_convs)):
                if i == 0:
                    outs.append(self.extra_convs[0](inputs[self.in_features[-1]]))
                else:
                    outs.append(self.extra_convs[i](outs[-1]))
        return outs
