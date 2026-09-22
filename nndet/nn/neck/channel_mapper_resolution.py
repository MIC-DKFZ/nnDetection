# Modifications licensed under:
# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
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


class ChannelMapper_resolution(nn.Module):
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
        Channel Mapper to return features of different resolution

        Args:
            conv: generator to create the convolution
            in_channels: list containing the number of channels of the backbone
                features (all feature maps)
            num_in_features: accepted for interface compatibility with the
                generic channel-mapper builder (which always passes the
                transformer's configured number of feature levels); unused
                here since this is a hardcoded 2-level channel mapper for
                Primus-style backbones (see ``in_channels`` below).
            kernel_size: Size of the convolution kernel for each scale.
            out_channels: Number of output channels for each scale.
            num_outs: (Optional) Number of output feature maps. There will be
                ``extra_convs`` when ``num_outs`` is larger than the length of
                ``in_features``. The last feature map (specified by
                ``num_in_features``) will be further processed by these
                convolutions and each convolution provides a new output.
            **kwargs: kwargs used by the convolution generator, could be
                'stride', 'groups', 'bias' or other
        """
        super(ChannelMapper_resolution, self).__init__()
        assert len(in_channels) == 2, (
            f"ChannelMapper_resolution hardcodes a 2-level topology (embed dim, "
            f"after first deconv), got {len(in_channels)} backbone feature maps"
        )

        self.convs = nn.ModuleList()
        padding = compute_padding_for_kernel(kernel_size)

        # hardcoded first version: in_channels [embed dim, after_first_deconv]
        # (e.g. [864, 112]) -> out_channels 128
        in_channel_conv = [in_channels[0], in_channels[0], out_channels, in_channels[1]]
        stride = [1,2,2,1]
        for i in range(len(in_channel_conv)):
            self.convs.append(
                conv(
                    in_channels=in_channel_conv[i],
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride[i],
                    **kwargs,
                )
            )



    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward function for the ChannelMapper to generate multifeature outputs

        Args:
            inputs: The backbone feature maps.

        Returns:
            tuple: A tuple of the processed features.
        """
        outs = []
        outs.append(self.convs[0](inputs[0]))
        outs.append(self.convs[1](inputs[0]))
        outs.append(self.convs[2](outs[-1]))
        outs.append(self.convs[3](inputs[1]))
        return outs
