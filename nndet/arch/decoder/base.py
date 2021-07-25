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
from typing import Sequence, List, Tuple, Union, Callable, Optional, TypeVar

from loguru import logger

from nndet.arch.conv import conv_kwargs_helper
from nndet.utils import to_dtype
from nndet.utils.info import experimental


class BaseUFPN(nn.Module):
    def __init__(self,
                 conv: Callable,
                 strides: Sequence[int],
                 in_channels: Sequence[int],
                 conv_kernels: Union[Sequence[Union[Sequence[int], int]], int],
                 decoder_levels: Union[Sequence[int], None],
                 fixed_out_channels: int,
                 min_out_channels: int = 8,
                 upsampling_mode: str = 'nearest',
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
            raise ValueError("Strides must contain same number of elements as channels.")
        if not len(in_channels) > 0:
            raise ValueError(f"Found unplausible channels {in_channels}")
        self.dim: int = conv.dim
        self.num_level = len(in_channels)
        self.in_channels = in_channels
        self.decoder_levels = decoder_levels

        # decoder and lateral convolutions
        self.strides = self.compute_stride_ratios(strides)
        self.conv_kernels, self.conv_paddings = self.determine_kernels_and_padding(conv_kernels)
        self.conv_settings = {
            "lateral": {"norm": norm_lateral, "activation": activation_lateral, "num": num_lateral},
            "out": {"norm": norm_out, "activation": activation_out, "num": num_out}
        }

        # upsampling layers
        self.strides = [to_dtype(stride, int) for stride in self.strides]
        self.upsampling_mode = upsampling_mode

        # additional information
        self.min_out_channels = min_out_channels
        self.fixed_out_channels = fixed_out_channels
        self.out_channels = self.compute_output_channels()

        self.lateral = nn.ModuleDict(
            {f"P{level}": self.get_lateral(conv, level) for level in range(self.num_level)}
        )
        self.out = nn.ModuleDict(
            {f"P{level}": self.get_conv(conv, level, "out") for level in range(self.num_level)}
        )
        self.up = nn.ModuleDict(
            {f"P{level}": self.get_up(conv, level) for level in range(1, self.num_level)}
        )

    def forward_lateral(self, inp_seq: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply lateral connections to incoming feature maps
        
        Args:
            inp_seq: sequence with feature maps (largest to samllest)

        Returns:
            List[Tensor]: resulting feature maps after lateral convolutions
        """
        return [self.lateral[f"P{level}"](fm) for level, fm in enumerate(inp_seq)]

    def forward_out(self, inp_seq: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply output convolutions to feature maps

        Args:
            inp_seq: sequence with feature maps (largest to smallest)

        Returns:
            List[Tensor]: resulting feature maps
        """
        return [self.out[f"P{level}"](fm) for level, fm in enumerate(inp_seq)]

    def compute_stride_ratios(self, strides) -> list:
        """
        Computes the strides between intermediate layers given the absolute stride

        Args:
            strides: absolute stride (stride with regard top highest resolution)
            dim: number of spatial dimensions

        Returns:
            List: compute strides between intermediate feature levels
        """
        strides = [stride if isinstance(stride, Sequence) else (stride, ) * self.dim for stride in strides]
        stride_ratios = []
        for i in range(1, len(strides)):
            stride_ratios.append(tuple(s1 / s0 for s1, s0 in zip(strides[i], strides[i - 1])))
        return stride_ratios

    def determine_kernels_and_padding(self, conv_kernels: Union[Sequence[Union[Sequence[int], int]], int]) -> \
            Tuple[List, List]:
        """
        Unify conv kernel input

        Args:
            conv_kernels: conv kernel to use for convolutions per level

        Returns:
            List: kernel sizes which can be passed directly to torch conv
            List: padding sizes which can be passed directly to torch conv
        """
        num_levels = len(self.in_channels)
        if isinstance(conv_kernels, int):
            _conv_paddings = [(conv_kernels - 1) // 2] * num_levels
            _conv_kernels = [conv_kernels] * num_levels
        elif isinstance(conv_kernels, Sequence):
            _conv_kernels = []
            _conv_paddings = []
            if not len(conv_kernels) == num_levels:
                raise ValueError(f"If conv kernels is not an integer it needs to be define the "
                                 f"kernel size for every level. Only found {len(conv_kernels)} "
                                 f"kernels und {num_levels} levels")
            for ck in conv_kernels:
                if isinstance(ck, int):
                    ck = [ck] * self.dim
                padding = [(i - 1) // 2 for i in ck]
                _conv_kernels.append(tuple(ck))
                _conv_paddings.append(tuple(padding))
        else:
            raise ValueError(f"{conv_kernels} is not a valid value of conv kernels in FPN")
        assert len(_conv_kernels) == num_levels
        assert len(_conv_paddings) == num_levels
        return _conv_kernels, _conv_paddings

    def compute_output_channels(self) -> List[int]:
        """
        Compute number of output channels

        Returns:
            List[int]: number of output channels for each level
        """
        out_channels = [self.fixed_out_channels] * self.num_level

        if self.decoder_levels is not None:
            ouput_levels = list(range(self.num_level))
            # filter for levels above decoder levels
            ouput_levels = [ol for ol in ouput_levels if ol < min(self.decoder_levels)]
            assert max(ouput_levels) < min(self.decoder_levels), "Can not decrease channels below decoder level"
            for ol in ouput_levels[::-1]:
                oc = max(self.min_out_channels, out_channels[ol + 1] // 2)
                out_channels[ol] = oc
        return out_channels

    def _get_kwargs(self, t: str) -> dict:
        """
        Create settings for respective conv type

        Args:
            t: define conv type. By default `lateral`, `fusion` or `out`

        Returns:
            dict: keyword arguments to pass to conv generator
        """
        return conv_kwargs_helper(
            norm=self.conv_settings[t]["norm"],
            activation=self.conv_settings[t]["activation"],
            )

    def get_lateral(self, conv: Callable, level: int) -> nn.Module:
        """
        Build a lateral convolution inside the fpn

        Args:
            conv: general convolution constructor
            level: level to build convolution for

        Returns:
            nn.Module: build connections
        """
        num = self.conv_settings["lateral"]["num"]
        _in_channels = [self.out_channels[level]] * num
        _in_channels[0] = self.in_channels[level]

        return torch.nn.Sequential(
            *[
                conv(_in_channels[i],
                     self.out_channels[level],
                     kernel_size=1,
                     padding=0,
                     stride=1,
                     **self._get_kwargs("lateral"),
                     )
                for i in range(num)]
        )

    def get_conv(self,
                 conv: Callable,
                 level: int,
                 name: str,
                 ) -> nn.Module:
        """
        Build a convolution inside the fpn

        Args:
            conv: general convolution constructor
            level: level to build convolution for
            name: type of convolution to look up configuration inside
                `self.conv_settings`

        Returns:
            nn.Module: build connections
        """
        return torch.nn.Sequential(
            *[
                conv(self.out_channels[level],
                     self.out_channels[level],
                     kernel_size=self.conv_kernels[level],
                     padding=self.conv_paddings[level],
                     stride=1,
                     **self._get_kwargs(name),
                     )
                for i in range(self.conv_settings[name]["num"])]
        )

    def get_up(self, conv: Callable, level: int):
        """
        Build a correctly configured upsampling block for the defined level

        Args:
            conv: base callable for convolutions
            level: number of level (fpn blocks)

        Returns:
            nn.Module: generated convolution
        """
        if self.upsampling_mode.lower() == 'transpose':
            up = conv(self.out_channels[level],
                      self.out_channels[level - 1],
                      kernel_size=self.strides[level - 1],
                      stride=self.strides[level - 1],
                      transposed=True,
                      add_norm=False,
                      add_act=False,
                      )
        else:
            up = torch.nn.Upsample(mode=self.upsampling_mode,
                                   scale_factor=self.strides[level - 1],
                                   )
            if not (self.out_channels[level] == self.out_channels[level - 1]):
                _conv = conv(self.out_channels[level],
                             self.out_channels[level - 1],
                             kernel_size=1, stride=1, padding=0,
                             add_norm=False,
                             add_act=False,
                             )
                up = torch.nn.Sequential(up, _conv)
        return up

    def get_channels(self) -> List[int]:
        """
        Return number of output channels

        Returns:
            List[int]: number of output channels for each image resolution
        """
        return self.out_channels


class UFPNModular(BaseUFPN):
    def __init__(self,
                 conv: Callable,
                 strides: Sequence[int],
                 in_channels: Sequence[int],
                 conv_kernels: Union[Sequence[Union[Sequence[int], int]], int],
                 decoder_levels: Union[Sequence[int], None],
                 fixed_out_channels: int,
                 min_out_channels: int = 8,
                 upsampling_mode: str = 'nearest',
                 num_lateral: int = 1,
                 norm_lateral: bool = False,
                 activation_lateral: bool = False,
                 num_out: int = 1,
                 norm_out: bool = False,
                 activation_out: bool = False,
                 num_fusion: int = 0,
                 norm_fusion: bool = False,
                 activation_fusion: bool = False,
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
            num_out: number of output convolutions
            norm_out: en-/disable normalization in output connections
            activation_out: en-/disable non linearity in out connections
            num_fusion: number of convolutions after elementwise addition of skip connections
            norm_fusion:  en-/disable normalization in fusion convolutions
            activation_fusion:  en-/disable non linearity in fusion convolutions
        """
        super().__init__(conv=conv, strides=strides, in_channels=in_channels,
                         conv_kernels=conv_kernels, decoder_levels=decoder_levels,
                         fixed_out_channels=fixed_out_channels,
                         min_out_channels=min_out_channels,
                         upsampling_mode=upsampling_mode,
                         num_lateral=num_lateral,
                         norm_lateral=norm_lateral,
                         activation_lateral=activation_lateral,
                         num_out=num_out,
                         norm_out=norm_out,
                         activation_out=activation_out,
                         )
        self.num_fusion = num_fusion
        self.conv_settings["fusion"] = {
            "norm": norm_fusion, "activation": activation_fusion, "num": num_fusion,
        }
        self.conv_settings["out"] = {
            "norm": norm_fusion, "activation": activation_fusion, "num": num_fusion,
        }

        if self.num_fusion > 0:
            self.fusion_bottom_up = nn.ModuleDict(
                {f"P{level}": self.get_conv(conv, level, "fusion") for level in range(self.num_level - 1)}
            )

    def forward(self, inp_seq: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass

        Args:
            inp_seq: sequence with feature maps (largest to samllest)

        Returns:
            List[Tensor]: resulting feature maps
        """
        fpn_maps = self.forward_lateral(inp_seq)

        # bottom up path way
        out_list = []  # sorted lowest to highest res
        for idx, x in enumerate(reversed(fpn_maps), 1):
            level = self.num_level - idx

            if idx != 1:
                x = x + up
                if self.num_fusion > 0:
                    x = self.fusion_bottom_up[f"P{level}"](x)

            if idx != self.num_level:
                up = self.up[f"P{level}"](x)

            out_list.append(x)
        return self.forward_out(reversed(out_list))


class PAUFPN(UFPNModular):
    @experimental
    def __init__(self,
                 conv: Callable,
                 strides: Sequence[int],
                 in_channels: Sequence[int],
                 conv_kernels: Union[Sequence[Union[Sequence[int], int]], int],
                 decoder_levels: Union[Sequence[int], None],
                 fixed_out_channels: int,
                 min_out_channels: int = 8,
                 upsampling_mode: str = 'nearest',
                 num_lateral: int = 1,
                 norm_lateral: bool = False,
                 activation_lateral: bool = False,
                 num_out: int = 1,
                 norm_out: bool = False,
                 activation_out: bool = False,
                 num_fusion: int = 1,
                 norm_fusion: bool = False,
                 activation_fusion: bool = False,
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
            num_out: number of output convolutions
            norm_out: en-/disable normalization in output connections
            activation_out: en-/disable non linearity in out connections
            num_fusion: number of convolutions after elementwise addition of skip connections
            norm_fusion:  en-/disable normalization in fusion convolutions
            activation_fusion:  en-/disable non linearity in fusion convolutions
        """
        super().__init__(conv=conv, strides=strides, in_channels=in_channels,
                         conv_kernels=conv_kernels, decoder_levels=decoder_levels,
                         fixed_out_channels=fixed_out_channels,
                         min_out_channels=min_out_channels,
                         upsampling_mode=upsampling_mode,
                         num_lateral=num_lateral,
                         norm_lateral=norm_lateral,
                         activation_lateral=activation_lateral,

                         # fpn out convs are not lateral connections towards pa layers
                         num_out=num_lateral,
                         norm_out=norm_lateral,
                         activation_out=activation_lateral,

                         num_fusion=num_fusion,
                         norm_fusion=norm_fusion,
                         activation_fusion=activation_fusion,
                         )

        self.conv_settings["pa_out"] = {"norm": norm_out, "activation": activation_out, "num": num_out}

        if self.num_fusion > 0:
            self.fusion_top_down = nn.ModuleDict(
                {f"N{level}": self.get_conv(conv, level, "fusion") for level in range(1, self.num_level)}
            )
        self.down = nn.ModuleDict(
            {f"N{level}": self.get_down(conv, level) for level in range(self.num_level - 1)},
        )
        self.pa_out = nn.ModuleDict(
            {f"N{level}": self.get_conv(conv, level, "pa_out") for level in range(self.num_level)}
        )

        logger.info(f"Building PAUFPN with lateral_kwargs {self._get_kwargs('lateral')}, "
                    f"fusion kwargs {self._get_kwargs('fusion')} and "
                    f"out_kwargs {self._get_kwargs('out')}")

    def get_down(self, conv: Callable, level: int) -> nn.Module:
        """
        Generate strided conv for downsampling

        Args:
            conv: base callable for convolutions
            level: number of level (fpn blocks)

        Returns:
            nn.Module: generated convolution
        """
        return conv(self.out_channels[level],
                    self.out_channels[level + 1],
                    kernel_size=self.conv_kernels[level],
                    padding=self.conv_paddings[level],
                    stride=self.strides[level],
                    add_norm=False,
                    add_act=False,
                    )

    def forward_out(self, inp_seq: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply output convolutions to feature maps

        Args:
            inp_seq: sequence with feature maps (largest to smallest)

        Returns:
            List[Tensor]: resulting feature maps
        """
        return [self.pa_out[f"N{level}"](fm) for level, fm in enumerate(inp_seq)]

    def forward(self, inp_seq: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass

        Args:
            inp_seq: sequence with feature maps (largest to samllest)

        Returns:
            List[Tensor]: resulting feature maps
        """
        fpn_maps = self.forward_lateral(inp_seq)

        # FPN
        intermediate = []  # sorted lowest to highest res
        for idx, x in enumerate(reversed(fpn_maps), 1):
            level = self.num_level - idx

            if idx != 1:
                x = x + up
                if self.num_fusion > 0:
                    x = self.fusion_bottom_up[f"P{level}"](x)

            if idx != self.num_level:
                up = self.up[f"P{level}"](x)

            intermediate.append(self.out[f"P{level}"](x))

        # PA
        out_list = []  # sorted highest to lowest res
        for level, x in enumerate(reversed(intermediate)):
            if level != 0:
                x = x + down
                if self.num_fusion > 0:
                    x = self.fusion_top_down[f"N{level}"](x)

            if level != self.num_level - 1:
                down = self.down[f"N{level}"](x)

            out_list.append(x)
        return self.forward_out(out_list)


DecoderType = TypeVar('DecoderType', bound=BaseUFPN)
