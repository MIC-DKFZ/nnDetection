from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn

from nndet.nn.layers.wrapper import compute_padding_for_kernel, torch_interpolation
from nndet.nn.neck.abstract import AbstractNeck
from nndet.utils.enums import InterpolationMode
from nndet.utils.typing import CONVGEN, ND_INT, ND_TUPLE_INT


class FPN(AbstractNeck):
    def __init__(
        self,
        conv: CONVGEN,
        in_channels: Sequence[int],
        relative_strides: Sequence[ND_TUPLE_INT],
        conv_kernels: ND_INT,
        first_decoder_level: int,
        last_decoder_level: int,
        fpn_out_channels: int,
        upsampling_mode: Union[str, InterpolationMode] = "nearest",
        num_lateral: int = 1,
        norm_lateral: bool = False,
        activation_lateral: bool = False,
        num_out: int = 1,
        norm_out: bool = False,
        activation_out: bool = False,
        num_fusion: int = 0,
        norm_fusion: bool = False,
        activation_fusion: bool = False,
        interpolation_kwargs: Optional[dict] = None,
    ):
        """
        Modular Implementation of Feature Pyramid Network (FPN)

        Overview::

            P0
            P1
            P2 - lateral -   X   - out ----------------------->
                            | upsample + optional conv fusion
            P3 - lateral -   X   - out ----------------------->
                            | upsample + optional conv fusion
            P4 - lateral -   X   - out ----------------------->
                            | upsample + optional conv fusion
            P5 - lateral -   X   - out ----------------------->

        Args:
            conv: convolution module to use internally
            in_channels: number of channels of each feature map
            relative_strides: define stride with respective to largest feature
                map (from lowest stride [highest res] to highest stride
                [lowest res])
            conv_kernels: define convolution kernels for decoder levels
            first_decoder_level: first decoder level (highest res), inclusive!
            last_decoder_level: last decoder level (lowest res), inclusive!
            fpn_out_channels: number of output channels (same for all FPN
                layers)
            upsampling_mode: can be one of `nearest` | `linear` | `cubic`
             or `transpose` to use transposed convolutions
            num_lateral: number of lateral convolutions
            norm_lateral: en-/disable normalization in lateral connections
            activation_lateral: en-/disable non linearity in lateral connections
            num_out: number of output convolutions
            norm_out: en-/disable normalization in output connections
            activation_out: en-/disable non linearity in output connections
            num_fusion: number of fusion convolutions. If 0 no fusion
                convolutions are used. Default 0.
            norm_fusion: en-/disable normalization in fusion connections
            activation_fusion: en-/disable non linearity in fusion connections
            interpolation_kwargs: interpolation kwargs, only used if the
                interpolation mode is not `transpose`
        """
        super().__init__()
        if len(relative_strides) != len(in_channels):
            raise ValueError(
                "Strides must contain same number of elements as channels."
            )
        if not len(in_channels) > 0:
            raise ValueError(f"Found unplausible channels {in_channels}")
        self.dim: int = conv.dim
        self.num_all_levels = len(in_channels)
        self.in_channels = in_channels
        self.relative_strides = relative_strides

        # decoder config
        self.fpn_out_channels = fpn_out_channels
        self.first_decoder_level = first_decoder_level
        self.last_decoder_level = last_decoder_level

        # lateral settings
        self.num_lateral = num_lateral
        self.norm_lateral = norm_lateral
        self.activation_lateral = activation_lateral

        # out settings
        self.num_out = num_out
        self.norm_out = norm_out
        self.activation_out = activation_out

        # fusion settings
        self.num_fusion = num_fusion
        self.norm_fusion = norm_fusion
        self.activation_fusion = activation_fusion

        # upsampling layers
        self.interpolation_mode = InterpolationMode(upsampling_mode)
        self.interpolation_kwargs = (
            {} if interpolation_kwargs is None else interpolation_kwargs
        )

        # create conv params
        self.conv_kernels = conv_kernels
        self.out_channels = self.compute_output_channels()

        # create convs
        self.lateral = nn.ModuleDict(
            {
                f"P{level}": self.build_lateral(conv, level)
                for level in range(
                    self.first_decoder_level, self.last_decoder_level + 1
                )
            }
        )
        self.out = nn.ModuleDict(
            {
                f"P{level}": self.build_out(conv, level)
                for level in range(
                    self.first_decoder_level, self.last_decoder_level + 1
                )
            }
        )
        self.up = nn.ModuleDict(  # first level doesn't need upsampling
            {
                f"P{level}": self.build_up(conv, level)
                for level in range(
                    self.first_decoder_level + 1, self.last_decoder_level + 1
                )
            }
        )
        if self.num_fusion > 0:
            self.fusion = nn.ModuleDict(  # last level doesn't need fusion
                {
                    f"P{level}": self.build_fusion(conv, level)
                    for level in range(
                        self.first_decoder_level, self.last_decoder_level
                    )
                }
            )
        else:
            self.fusion = None

    def compute_output_channels(self) -> List[int]:
        """
        Compute number of output channels

        Returns:
            List[int]: number of output channels for each level
        """
        out_channels = [
            None if level_idx < self.first_decoder_level else self.fpn_out_channels
            for level_idx in len(self.num_all_levels)
        ]
        return out_channels

    def get_channels(self) -> List[int]:
        """
        Compute number of channels for each returned feature map
        inside the forward pass

        Returns
            List[int]: list with number of channels corresponding to
                returned feature maps
        """
        return self.out_channels

    def build_lateral(
        self,
        conv: CONVGEN,
        level: int,
    ) -> nn.Module:
        """
        Build a lateral connection inside the FPN

        Args:
            conv: general convolution constructor
            level: level index

        Returns:
            nn.Module: build connections
        """
        lateral_connection = []
        for i in range(self.num_lateral):
            _in_channels = (
                self.in_channels[level] if i == 0 else self.out_channels[level]
            )

            lateral_connection.append(
                conv(
                    _in_channels,
                    self.out_channels[level],
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    add_norm=self.norm_lateral,
                    add_act=self.activation_lateral,
                )
            )
        return nn.Sequential(*lateral_connection)

    def build_up(
        self,
        conv: CONVGEN,
        level: int,
    ) -> nn.Module:
        """
        Build upsampling block.

        Args:
            conv: base callable for convolutions
            level: level index

        Returns:
            nn.Module: generated convolution
        """
        if self.interpolation_mode == InterpolationMode.TRANSPOSE:
            up = conv(
                self.out_channels[level],
                self.out_channels[level - 1],
                kernel_size=self.relative_strides[level],
                stride=self.relative_strides[level],
                transposed=True,
                add_norm=False,
                add_act=False,
            )
        else:
            up = nn.Upsample(
                mode=torch_interpolation(self.interpolation_mode, dim=self.dim),
                scale_factor=self.relative_strides[level],
                **self.interpolation_kwargs,
            )
        return up

    def build_out(
        self,
        conv: CONVGEN,
        level: int,
    ) -> nn.Module:
        """
        Build a output connection

        Args:
            conv: general convolution constructor
            level: level index

        Returns:
            nn.Module: build connections
        """
        kernel_size = self.conv_kernels[level]
        padding = compute_padding_for_kernel(kernel_size)

        return torch.nn.Sequential(
            *[
                conv(
                    self.out_channels[level],
                    self.out_channels[level],
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=1,
                    add_norm=self.norm_out,
                    add_act=self.activation_out,
                )
                for _ in range(self.num_out)
            ]
        )

    def build_fusion(
        self,
        conv: CONVGEN,
        level: int,
    ) -> nn.Module:
        """
        Build a fusion connection

        Args:
            conv: general convolution constructor
            level: level index

        Returns:
            nn.Module: build connections
        """
        kernel_size = self.conv_kernels[level]
        padding = compute_padding_for_kernel(kernel_size)

        return torch.nn.Sequential(
            *[
                conv(
                    self.out_channels[level],
                    self.out_channels[level],
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=1,
                    add_norm=self.norm_fusion,
                    add_act=self.activation_fusion,
                )
                for _ in range(self.num_fusion)
            ]
        )

    def forward(
        self,
        backbone_output: Sequence[torch.Tensor],
    ) -> List[Optional[torch.Tensor]]:
        """
        Forward pass

        Args:
            inp_seq: sequence with feature maps. Sorted from P0 (highest res)
                to PX (lowest res).

        Returns:
            List[Tensor]: resulting feature maps. Sorted by P0 (highest res)
                to PX (lowest res). Level which were not processed by
                this network contain None to prevent accidential mixup
                of unsupported network components!
        """
        out_list = [None for i in range(len(backbone_output))]

        for level_idx in range(
            self.last_decoder_level, self.first_decoder_level - 1, -1
        ):
            # iterate from last to first level
            lateral_feat = self.lateral[f"P{level_idx}"](backbone_output[level_idx])

            if level_idx != self.last_decoder_level:
                comb_feat = lateral_feat + up  # noqa: F821
                if self.fusion is not None:
                    comb_feat = self.fusion[f"P{level_idx}"](comb_feat)
            else:
                comb_feat = lateral_feat  # last level does not run through combination

            if level_idx != self.first_decoder_level:
                up = self.up[f"P{level_idx}"](comb_feat)  # noqa: F841

            out_list[level_idx] = self.out[f"P{level_idx}"](comb_feat)
        return out_list


class UFPN(FPN):
    def __init__(
        self,
        conv: CONVGEN,
        in_channels: Sequence[int],
        relative_strides: Sequence[ND_TUPLE_INT],
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
        num_fusion: int = 0,
        norm_fusion: bool = False,
        activation_fusion: bool = False,
        min_out_channels: int = 8,
        reduction_out_channels: int = 2,
        interpolation_kwargs: Optional[dict] = None,
    ):
        """
        Modular Implementation of Feature Pyramid Network (FPN)

        Overview::

            P1 - lateral -  X/4  - out/4 --------------------->
                            | upsample + optional conv fusion
            P1 - lateral -  X/2  - out/2 --------------------->
                            | upsample + optional conv fusion
            P2 - lateral -   X   - out ----------------------->
                            | upsample + optional conv fusion
            P3 - lateral -   X   - out ----------------------->
                            | upsample + optional conv fusion
            P4 - lateral -   X   - out ----------------------->
                            | upsample + optional conv fusion
            P5 - lateral -   X   - out ----------------------->

        Reduction set to two in this example.

        Args:
            conv: convolution module to use internally
            in_channels: number of channels of each feature map
            relative_strides: define stride with respective to largest feature
                map (from lowest stride [highest res] to highest stride
                [lowest res])
            conv_kernels: define convolution kernels for decoder levels
            first_decoder_level: first decoder level (highest res), inclusive!
            last_decoder_level: last decoder level (lowest res), inclusive!
            fpn_out_channels: number of output channels (same for all FPN
                layers)
            upsampling_mode: if `transpose` a transposed convolution is used
                for upsampling, otherwise it defines the method used in
                `torch.interpolate`
            num_lateral: number of lateral convolutions
            norm_lateral: en-/disable normalization in lateral connections
            activation_lateral: en-/disable non linearity in lateral connections
            num_out: number of output convolutions
            norm_out: en-/disable normalization in output connections
            activation_out: en-/disable non linearity in output connections
            num_fusion: number of fusion convolutions. If 0 no fusion
                convolutions are used. Default 0.
            norm_fusion: en-/disable normalization in fusion connections
            activation_fusion: en-/disable non linearity in fusion connections
        """
        self.first_reduction_decoder_level = first_decoder_level
        self.min_out_channels = min_out_channels
        self.reduction_out_channels = reduction_out_channels
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            relative_strides=relative_strides,
            conv_kernels=conv_kernels,
            first_decoder_level=0,  # need to build and forward all levels
            last_decoder_level=last_decoder_level,
            fpn_out_channels=fpn_out_channels,
            upsampling_mode=upsampling_mode,
            num_lateral=num_lateral,
            norm_lateral=norm_lateral,
            activation_lateral=activation_lateral,
            num_out=num_out,
            norm_out=norm_out,
            activation_out=activation_out,
            num_fusion=num_fusion,
            norm_fusion=norm_fusion,
            activation_fusion=activation_fusion,
            interpolation_kwargs=interpolation_kwargs,
        )

    def compute_output_channels(self) -> List[int]:
        """
        Compute number of output channels
        Channels starting between first_decoder_level and last_decoder_level
        have the same number of channels. Channels below

        Returns:
            List[int]: number of output channels for each level
        """
        out_channels = [self.fpn_out_channels for _ in range(self.num_all_levels)]

        for i in range(0, self.first_reduction_decoder_level):
            _diff = self.first_reduction_decoder_level - i
            assert _diff > 0
            _out_channels_reduced = int(
                self.fpn_out_channels / (self.reduction_out_channels**_diff)
            )
            out_channels[i] = max(_out_channels_reduced, self.min_out_channels)
        return out_channels

    def build_up(
        self,
        conv: CONVGEN,
        level: int,
    ) -> nn.Module:
        """
        Build upsampling block.

        Args:
            conv: base callable for convolutions
            level: level index

        Returns:
            nn.Module: generated convolution
        """
        if self.interpolation_mode == InterpolationMode.TRANSPOSE:
            up = conv(
                self.out_channels[level],
                self.out_channels[level - 1],
                kernel_size=self.relative_strides[level],
                stride=self.relative_strides[level],
                transposed=True,
                add_norm=False,
                add_act=False,
            )
        else:
            up = torch.nn.Upsample(
                mode=torch_interpolation(self.interpolation_mode, dim=self.dim),
                scale_factor=self.relative_strides[level],
                **self.interpolation_kwargs,
            )
            if not (self.out_channels[level] == self.out_channels[level - 1]):
                _conv = conv(
                    self.out_channels[level],
                    self.out_channels[level - 1],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    add_norm=False,
                    add_act=False,
                )
                up = torch.nn.Sequential(up, _conv)
        return up
