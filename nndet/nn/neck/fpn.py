# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

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
        upsampling_mode: Union[str, InterpolationMode] = "linear",
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
        **kwargs,
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
                interpolation mode is not `transpose`. If None, `align_corners`
                will be set to `True` per default.
            kwargs: enables usage of U-Like configs without modification,
                not used in this module
        """
        super().__init__()
        if len(relative_strides) != len(in_channels):
            raise ValueError("Strides must contain same number of elements as channels.")
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
        self.interpolation_kwargs = {"align_corners": True} if interpolation_kwargs is None else interpolation_kwargs

        # create conv params
        self.conv_kernels = conv_kernels
        self.out_channels = self.compute_output_channels()

        # create convs
        self.lateral = nn.ModuleDict(
            {
                f"P{level}": self.build_lateral(conv, level)
                for level in range(self.first_decoder_level, self.last_decoder_level + 1)
            }
        )
        self.out = nn.ModuleDict(
            {
                f"P{level}": self.build_out(conv, level)
                for level in range(self.first_decoder_level, self.last_decoder_level + 1)
            }
        )
        self.up = nn.ModuleDict(  # first level doesn't need upsampling
            {
                f"P{level}": self.build_up(conv, level)
                for level in range(self.first_decoder_level + 1, self.last_decoder_level + 1)
            }
        )
        if self.num_fusion > 0:
            self.fusion = nn.ModuleDict(  # last level doesn't need fusion
                {
                    f"P{level}": self.build_fusion(conv, level)
                    for level in range(self.first_decoder_level, self.last_decoder_level)
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
            None
            if level_idx < self.first_decoder_level or level_idx > self.last_decoder_level
            else self.fpn_out_channels
            for level_idx in range(self.num_all_levels)
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
            _in_channels = self.in_channels[level] if i == 0 else self.out_channels[level]

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
                bias=False,
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

        for level_idx in range(self.last_decoder_level, self.first_decoder_level - 1, -1):
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
        upsampling_mode: str = "linear",
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
            _out_channels_reduced = int(self.fpn_out_channels / (self.reduction_out_channels**_diff))
            out_channels[i] = max(_out_channels_reduced, self.min_out_channels)

        for i in range(self.last_decoder_level + 1, self.num_all_levels):
            out_channels[i] = None
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
                bias=False,
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


class UpFPN(FPN):
    def __init__(
        self,
        conv: CONVGEN,
        in_channels: Sequence[int],
        relative_strides: Sequence[ND_TUPLE_INT],
        conv_kernels: ND_INT,
        first_decoder_level: int,
        last_decoder_level: int,
        fpn_out_channels: int,
        upsampling_mode: Union[str, InterpolationMode] = "linear",
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
        num_up: int = 2,
        norm_up: bool = True,
        activation_up: bool = True,
        reduction_up: float = 4.0,
        up_interpolation_mode: Union[str, InterpolationMode] = "linear",
        up_interpolation_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Modular Implementation of Feature Pyramid Network (FPN)

        Overview::

            P0                      |   - upsample to stride 1
                                    |
            P1                      |
                                    | optional up convs added here
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
            num_up: additional convs insert before upsampling. If 0 no convs
                will be inserted. At least on up conv is needed to
                reduce number of channels.
            norm_up: add normalisation layers to additional convs
            activation_up: add activations to additional convs
            up_interpolation_mode: interpolation mode used to upsample to
                stride 1. Transpose Mode is not supported here.
            up_interpolation_kwargs: Passed to interpolation function for
                stride 1 upsampling. If None, `align_corners=True` will be
                set by default.
            kwargs: enables usage of U-Like configs without modification,
                not used in this module
        """
        self.reduction_up = reduction_up
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            relative_strides=relative_strides,
            conv_kernels=conv_kernels,
            first_decoder_level=first_decoder_level,
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
        # up convs
        self.num_up = num_up
        self.norm_up = norm_up
        self.activation_up = activation_up

        if self.num_up < 1:
            raise ValueError("At least one up conv is required.")

        # interpolation
        self.up_interpolation_mode = InterpolationMode(up_interpolation_mode)
        if self.up_interpolation_mode == InterpolationMode.TRANSPOSE:
            raise ValueError("Stride one upsampling does not support transpose mode.")
        self.up_interpolation_kwargs = (
            {"align_corners": True} if up_interpolation_kwargs is None else up_interpolation_kwargs
        )
        self.up_stride_one = self.build_stride_one(conv)

    def build_stride_one(self, conv) -> nn.Module:
        """
        Build upsampling block to interpolate first decoder level to stride one.

        Returns:
            nn.Module: Upsampling module
        """
        up_modules = []
        reduced_channels = int(self.fpn_out_channels / self.reduction_up)

        for i in range(self.num_up):
            _in_channels = self.fpn_out_channels if i == 0 else reduced_channels
            up_modules.append(
                conv(
                    _in_channels,
                    reduced_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    add_norm=self.norm_up,
                    add_act=self.activation_up,
                )
            )

        scale_factor = [1 for _ in range(self.dim)]
        for level_idx in range(self.first_decoder_level + 1):
            _relative_level_stride = self.relative_strides[level_idx]
            if _relative_level_stride is not None:  # account for skipped levels
                if not isinstance(_relative_level_stride, Sequence):
                    scale_factor = [s * _relative_level_stride for s in scale_factor]
                else:
                    assert len(_relative_level_stride) == len(scale_factor)
                    scale_factor = [s * r for s, r in zip(scale_factor, _relative_level_stride)]

        up_modules.append(
            nn.Upsample(
                mode=torch_interpolation(self.up_interpolation_mode, dim=self.dim),
                scale_factor=tuple(scale_factor),
                **self.up_interpolation_kwargs,
            )
        )
        return torch.nn.Sequential(*up_modules)

    def compute_output_channels(self) -> List[int]:
        """
        Compute number of output channels

        Returns:
            List[int]: number of output channels for each level
        """
        out_channels = super().compute_output_channels()
        out_channels[0] = int(self.fpn_out_channels / self.reduction_up)
        return out_channels

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
        fpn_output = super().forward(backbone_output=backbone_output)
        fpn_output[0] = self.up_stride_one(fpn_output[self.first_decoder_level])
        return fpn_output
