from typing import Sequence

import torch.nn as nn

from nndet.nn.neck.abstract import AbstractNeck
from nndet.utils import to_dtype
from nndet.utils.typing import CONVGEN, ND_INT


class FPN(AbstractNeck):
    def __init__(
        self,
        conv: CONVGEN,
        in_channels: Sequence[int],
        strides: Sequence[Sequence[int]],
        conv_kernels: ND_INT,
        first_decoder_level: int,
        last_decoder_level: int,
        fpn_out_channels: int,
        upsampling_mode: str = "nearest",
        num_lateral: int = 1,
        norm_lateral: bool = False,
        activation_lateral: bool = False,
        num_out: int = 1,
        norm_out: bool = False,
        activation_out: bool = False,
    ):
        """
        Base class for UFPN like builds
        Just overwrite `compute_output_channels` to generate different
        output channels

        Args:
            conv: convolution module to use internally
            strides: define stride with respective to largest feature map
                (from lowest stride [highest res] to highest stride [lowest res])
            in_channels: number of channels of each feature maps
            conv_kernels: define convolution kernels for decoder levels
            decoder_levels: levels which are later used for detection.
                If None a normal fpn is used.
            fixed_out_channels: number of output channels in fixed layers
            min_out_channels: minimum number of feature channels for
                layers above decoder levels
            upsampling_mode: if `transpose` a transposed convolution is used
                for upsampling, otherwise it defines the method used in
                torch.interpolate followed by a 1x1 convolution to adjust
                the channels
            num_lateral: number of lateral convolutions
            norm_lateral: en-/disable normalization in lateral connections
            activation_lateral: en-/disable non linearity in lateral connections
        """
        super().__init__()
        if len(strides) != len(in_channels):
            raise ValueError(
                "Strides must contain same number of elements as channels."
            )
        if not len(in_channels) > 0:
            raise ValueError(f"Found unplausible channels {in_channels}")
        self.dim: int = conv.dim
        self.num_level = len(in_channels)
        self.in_channels = in_channels

        # decoder config
        self.fpn_out_channels = fpn_out_channels
        self.first_decoder_level = first_decoder_level
        self.last_decoder_level = last_decoder_level

        # lateral settings
        self.norm_lateral = norm_lateral
        self.activation_lateral = activation_lateral
        self.num_lateral = num_lateral

        # out settings
        self.norm_out = norm_out
        self.activation_out = activation_out
        self.num_out = num_out

        # upsampling layers
        self.strides = [to_dtype(stride, int) for stride in self.strides]
        self.upsampling_mode = upsampling_mode

        # create conv params
        self.strides = self.compute_stride_ratios(strides)
        self.conv_kernels, self.conv_paddings = self.determine_kernels_and_padding(
            conv_kernels
        )
        self.out_channels = self.compute_output_channels()

        # create convs
        self.lateral = nn.ModuleDict(
            {
                f"P{level}": self.get_lateral(conv, level)
                for level in range(self.num_level)
            }
        )
        self.out = nn.ModuleDict(
            {
                f"P{level}": self.get_conv(conv, level, "out")
                for level in range(self.num_level)
            }
        )
        self.up = nn.ModuleDict(
            {
                f"P{level}": self.get_up(conv, level)
                for level in range(1, self.num_level)
            }
        )


class UFPN(FPN):
    pass
    # min_out_channels: int = 8,
    # self.min_out_channels = min_out_channels
