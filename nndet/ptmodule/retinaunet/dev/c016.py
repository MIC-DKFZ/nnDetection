# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Type

from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.backbone.blueprints.conv import ConvBackbone
from nndet.nn.backbone.blueprints.resconv import ResConvBackbone
from nndet.nn.heads.classifier import FocalClassifier
from nndet.nn.heads.comb import BoxHeadAll, BoxHeadHNM
from nndet.nn.heads.regressor import L1Regressor
from nndet.nn.layers.conv import ConvGroupLReLU, ConvInstanceLReLU
from nndet.nn.neck.abstract import AbstractNeck
from nndet.nn.neck.fpn import UFPN, UpFPN
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.retinaunet.runv001 import RetinaUNetCV001Focal, RetinaUNetV001


@MODULE_REGISTRY.register
class RetinaUNetC016(RetinaUNetV001):
    backbone_cls: Type[AbstractBackbone] = ConvBackbone
    backbone_conv_cls = ConvInstanceLReLU

    neck_cls: Type[AbstractNeck] = UFPN
    neck_conv_cls = ConvInstanceLReLU

    head_cls = BoxHeadHNM
    head_conv_cls = ConvGroupLReLU
    head_regressor_cls = L1Regressor


@MODULE_REGISTRY.register
class RetinaUNetC016Up(RetinaUNetC016):
    neck_cls: Type[AbstractNeck] = UpFPN


@MODULE_REGISTRY.register
class RetinaUNetC016Res(RetinaUNetC016):
    backbone_cls: Type[AbstractBackbone] = ResConvBackbone


@MODULE_REGISTRY.register
class RetinaUNetC016Focal(RetinaUNetCV001Focal):
    backbone_cls: Type[AbstractBackbone] = ConvBackbone
    backbone_conv_cls = ConvInstanceLReLU

    neck_cls: Type[AbstractNeck] = UFPN
    neck_conv_cls = ConvInstanceLReLU

    head_cls = BoxHeadAll
    head_conv_cls = ConvGroupLReLU
    head_sampler_cls = None
    head_regressor_cls = L1Regressor
    head_classifier_cls = FocalClassifier



from typing import Callable, List, Sequence, Tuple, TypeVar, Union

import torch
import torch.nn as nn
from loguru import logger

from nndet.utils import to_dtype
from nndet.utils.info import experimental
from nndet.nn.layers.wrapper import Generator


def conv_kwargs_helper(norm: bool, activation: bool):
    """
    Helper to force disable normalization and activation in layers
    which have those by default

    Args:
        norm: en-/disable normalization layer
        activation: en-/disable activation layer

    Returns:
        dict: keyword arguments to pass to conv generator
    """
    kwargs = {
        "add_norm": norm,
        "add_act": activation,
    }
    return kwargs


class BaseUFPN(nn.Module):
    def __init__(
        self,
        conv: Callable,
        strides: Sequence[int],
        in_channels: Sequence[int],
        conv_kernels: Union[Sequence[Union[Sequence[int], int]], int],
        decoder_levels: Union[Sequence[int], None],
        fixed_out_channels: int,
        min_out_channels: int = 8,
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
        self.decoder_levels = decoder_levels

        # decoder and lateral convolutions
        self.strides = self.compute_stride_ratios(strides)
        self.conv_kernels, self.conv_paddings = self.determine_kernels_and_padding(
            conv_kernels
        )
        self.conv_settings = {
            "lateral": {
                "norm": norm_lateral,
                "activation": activation_lateral,
                "num": num_lateral,
            },
            "out": {"norm": norm_out, "activation": activation_out, "num": num_out},
        }

        # upsampling layers
        self.strides = [to_dtype(stride, int) for stride in self.strides]
        self.upsampling_mode = upsampling_mode

        # additional information
        self.min_out_channels = min_out_channels
        self.fixed_out_channels = fixed_out_channels
        self.out_channels = self.compute_output_channels()

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
        strides = [
            stride if isinstance(stride, Sequence) else (stride,) * self.dim
            for stride in strides
        ]
        stride_ratios = []
        for i in range(1, len(strides)):
            stride_ratios.append(
                tuple(s1 / s0 for s1, s0 in zip(strides[i], strides[i - 1]))
            )
        return stride_ratios

    def determine_kernels_and_padding(
        self, conv_kernels: Union[Sequence[Union[Sequence[int], int]], int]
    ) -> Tuple[List, List]:
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
                raise ValueError(
                    f"If conv kernels is not an integer it needs to be define the "
                    f"kernel size for every level. Only found {len(conv_kernels)} "
                    f"kernels und {num_levels} levels"
                )
            for ck in conv_kernels:
                if isinstance(ck, int):
                    ck = [ck] * self.dim
                padding = [(i - 1) // 2 for i in ck]
                _conv_kernels.append(tuple(ck))
                _conv_paddings.append(tuple(padding))
        else:
            raise ValueError(
                f"{conv_kernels} is not a valid value of conv kernels in FPN"
            )
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
            assert max(ouput_levels) < min(
                self.decoder_levels
            ), "Can not decrease channels below decoder level"
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
                conv(
                    _in_channels[i],
                    self.out_channels[level],
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    **self._get_kwargs("lateral"),
                )
                for i in range(num)
            ]
        )

    def get_conv(
        self,
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
                conv(
                    self.out_channels[level],
                    self.out_channels[level],
                    kernel_size=self.conv_kernels[level],
                    padding=self.conv_paddings[level],
                    stride=1,
                    **self._get_kwargs(name),
                )
                for i in range(self.conv_settings[name]["num"])
            ]
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
        if self.upsampling_mode.lower() == "transpose":
            up = conv(
                self.out_channels[level],
                self.out_channels[level - 1],
                kernel_size=self.strides[level - 1],
                stride=self.strides[level - 1],
                transposed=True,
                add_norm=False,
                add_act=False,
            )
        else:
            up = torch.nn.Upsample(
                mode=self.upsampling_mode,
                scale_factor=self.strides[level - 1],
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

    def get_channels(self) -> List[int]:
        """
        Return number of output channels

        Returns:
            List[int]: number of output channels for each image resolution
        """
        return self.out_channels


class UFPNModular(BaseUFPN):
    def __init__(
        self,
        conv: Callable,
        strides: Sequence[int],
        in_channels: Sequence[int],
        conv_kernels: Union[Sequence[Union[Sequence[int], int]], int],
        decoder_levels: Union[Sequence[int], None],
        fixed_out_channels: int,
        min_out_channels: int = 8,
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
        super().__init__(
            conv=conv,
            strides=strides,
            in_channels=in_channels,
            conv_kernels=conv_kernels,
            decoder_levels=decoder_levels,
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
            "norm": norm_fusion,
            "activation": activation_fusion,
            "num": num_fusion,
        }
        self.conv_settings["out"] = {
            "norm": norm_fusion,
            "activation": activation_fusion,
            "num": num_fusion,
        }

        if self.num_fusion > 0:
            self.fusion_bottom_up = nn.ModuleDict(
                {
                    f"P{level}": self.get_conv(conv, level, "fusion")
                    for level in range(self.num_level - 1)
                }
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
                x = x + up  # noqa: F821
                if self.num_fusion > 0:
                    x = self.fusion_bottom_up[f"P{level}"](x)

            if idx != self.num_level:
                up = self.up[f"P{level}"](x)  # noqa: F841

            out_list.append(x)
        return self.forward_out(reversed(out_list))


@MODULE_REGISTRY.register
class RetinaUNetC016OldUFPN(RetinaUNetV001):
    neck_cls: Type[AbstractNeck] = UFPNModular

    @classmethod
    def _build_neck(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        backbone,
    ):
        """
        Build neck network

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            DecoderType: neck instance
        """
        conv = Generator(cls.neck_conv_cls, plan_arch["dim"])
        logger.info(
            f"Building:: neck {cls.neck_cls.__name__}: {model_cfg['neck_kwargs']}"
        )
        neck = cls.neck_cls(
            conv=conv,
            conv_kernels=plan_arch["conv_kernels"],
            strides=backbone.get_absolute_strides(),
            in_channels=backbone.get_channels(),
            decoder_levels=plan_arch["decoder_levels"],
            fixed_out_channels=plan_arch["fpn_channels"],
            **model_cfg["neck_kwargs"],
        )
        return neck


from abc import abstractmethod
from typing import Dict, List, TypeVar, Union

import torch
import torch.nn as nn

__all__ = ["AbstractEncoder"]


class AbstractEncoder(nn.Module):
    def __int__(self, **kwargs):
        """
        Provides an abstract interface for backbone networks
        """
        super().__init__(**kwargs)

    @abstractmethod
    def forward(self, x) -> List[torch.Tensor]:
        """
        Forward input through network

        Args
            x (torch.tensor): input tensor

        Returns
            list: list with feature maps from multiple resolutions
        """
        raise NotImplementedError

    @abstractmethod
    def get_channels(self) -> List[int]:
        """
        Compute number of channels for each returned feature map
        inside the forward pass

        Returns
            List[int]: list with number of channels corresponding to
                returned feature maps
        """
        raise NotImplementedError

    @abstractmethod
    def get_strides(self) -> List[Dict[str, Union[List[int], int]]]:
        """
        Compute number backbone strides for 2d and 3d case and all options
        of network

        Returns
            List[Dict[str, Union[List[int], int]]]: dict with 'xy' for 2d
                stride and optional 'z' for 3d cases. List
                describes stride at respective output level
        """
        raise NotImplementedError


from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

__all__ = ["Encoder"]


class Encoder(AbstractEncoder):
    def __init__(
        self,
        conv: Callable[[], nn.Module],
        conv_kernels: Sequence[Union[Tuple[int], int]],
        strides: Sequence[Union[Tuple[int], int]],
        block_cls,
        in_channels: int,
        start_channels: int,
        stage_kwargs: Sequence[dict] = None,
        out_stages: Sequence[int] = None,
        max_channels: int = None,
        first_block_cls = None,
    ):
        """
        Build a modular encoder model with specified blocks
        The Encoder consists of "stages" which (in general) represent one
        resolution in the resolution pyramid. The first level alwasys has
        full resolution.

        Args:
            conv: conv generator to use for internal convolutions
            strides: strides for pooling layers. Should have one
                element less than conv_kernels
            conv_kernels: kernel sizes for convolutions
            block_cls: generate a block of convolutions (
                e.g. stacked residual blocks)
            in_channels: number of input channels
            start_channels: number of start channels
            stage_kwargs: additional keyword arguments for stages.
                Defaults to None.
            out_stages: define which stages should be returned. If `None` all
                stages will be returned.Defaults to None.
            first_block_cls: generate a block of convolutions for the first stage
                By default this equal the provided block_cls
        """
        super().__init__()
        self.num_stages = len(conv_kernels)
        self.dim = conv.dim
        if stage_kwargs is None:
            stage_kwargs = [{}] * self.num_stages
        elif isinstance(stage_kwargs, dict):
            stage_kwargs = [stage_kwargs] * self.num_stages
        assert len(stage_kwargs) == len(conv_kernels)

        if out_stages is None:
            self.out_stages = list(range(self.num_stages))
        else:
            self.out_stages = out_stages
        if first_block_cls is None:
            first_block_cls = block_cls

        stages = []
        self.out_channels = []
        if isinstance(strides[0], int):
            strides = [tuple([s] * self.dim) for s in strides]
        self.strides = strides
        for stage_id in range(self.num_stages):
            if stage_id == 0:
                _block = first_block_cls(
                    conv=conv,
                    in_channels=in_channels,
                    out_channels=start_channels,
                    conv_kernel=conv_kernels[stage_id],
                    stride=None,
                    max_out_channels=max_channels,
                    **stage_kwargs[stage_id],
                )
            else:
                _block = block_cls(
                    conv=conv,
                    in_channels=in_channels,
                    out_channels=None,
                    conv_kernel=conv_kernels[stage_id],
                    stride=strides[stage_id - 1],
                    max_out_channels=max_channels,
                    **stage_kwargs[stage_id],
                )
            in_channels = _block.get_output_channels()
            self.out_channels.append(in_channels)
            stages.append(_block)
        self.stages = torch.nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward data through encoder

        Args:
            x: input data

        Returns:
            List[torch.Tensor]: list of output from stages defined by
                param:`out_stages`
        """
        outputs = []
        for stage_id, module in enumerate(self.stages):
            x = module(x)
            if stage_id in self.out_stages:
                outputs.append(x)
        return outputs

    def get_channels(self) -> List[int]:
        """
        Compute number of channels for each returned feature map inside the forward pass

        Returns
            list: list with number of channels corresponding to returned feature maps
        """
        out_channels = []
        for stage_id in range(self.num_stages):
            if stage_id in self.out_stages:
                out_channels.append(self.out_channels[stage_id])
        return out_channels

    def get_relative_strides(self) -> List[List[int]]:
        """
        Compute number backbone strides for 2d and 3d case and all options of network

        Returns
            List[List[int]]: defines the absolute stride for each output
                feature map with respect to input size
        """
        out_strides = []
        for stage_id in range(self.num_stages):
            if stage_id == 0:
                out_strides.append([1] * self.dim)
            else:
                out_strides.append(self.strides[stage_id - 1])
        return out_strides


from abc import abstractmethod
from functools import reduce
from typing import Callable, Sequence, Tuple, Union

import torch
import torch.nn as nn

from nndet.utils.typing import ND_INT


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

    def __init__(
        self,
        conv: Callable[[], nn.Module],
        in_channels: int,
        conv_kernel: ND_INT,
        stride: ND_INT = None,
        out_channels: int = None,
        max_out_channels: int = None,
        num_blocks: int = 1,
        **kwargs,
    ):
        """
        Plain stack of convolutions. Strides > 1 are applied at the beginning
        by a strided convolution and the first convolution raises the number of
        channels to `out_channels`.

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
        super().__init__(out_channels=None)  # out_channels will be overwritten later
        if (
            out_channels is not None
            and max_out_channels is not None
            and out_channels > max_out_channels
        ):
            raise ValueError(
                "Output channels can not be larger" "than max output channels"
            )
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
        _convs.append(
            self.build_block(
                conv=conv,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=conv_kernel,
                stride=stride,
                padding=padding,
                **kwargs,
            )
        )
        for _ in range(num_blocks - 1):
            _convs.append(
                self.build_block(
                    conv=conv,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=conv_kernel,
                    stride=1,
                    padding=padding,
                    **kwargs,
                )
            )

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
    def build_block(
        self,
        conv: Callable[[], nn.Module],
        in_channels: int,
        out_channels: int,
        kernel_size: ND_INT,
        stride: ND_INT,
        padding: ND_INT,
    ) -> nn.Module:
        raise NotImplementedError


class StackedConvBlock2(StackedBlock):
    def build_block(
        self,
        conv: Callable,
        in_channels: int,
        out_channels: int,
        kernel_size: ND_INT,
        stride: ND_INT,
        padding: ND_INT,
        **kwargs,
    ) -> nn.Module:
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
            conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                **kwargs,
            ),
            conv(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                **kwargs,
            ),
        )


import copy

@MODULE_REGISTRY.register
class RetinaUNetC016OldEncoder(RetinaUNetV001):
    backbone_cls: Type[AbstractBackbone] = Encoder
    backbone_block = StackedConvBlock2

    @classmethod
    def _build_backbone(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        **kwargs,
    ):
        """
        Build backbone network

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            EncoderType: backbone instance
        """
        conv = Generator(cls.backbone_conv_cls, plan_arch["dim"])

        logger.info(
            f"Building:: backbone {cls.backbone_cls.__name__}: {model_cfg['backbone_kwargs']} "
        )

        _kwargs = copy.deepcopy(model_cfg["backbone_kwargs"])
        if "max_channels" in _kwargs:
            max_channels = _kwargs.pop("max_channels")
        else:
            max_channels = plan_arch.get("max_channels", 320)

        backbone = cls.backbone_cls(
            conv=conv,
            conv_kernels=plan_arch["conv_kernels"],
            strides=plan_arch["strides"],
            block_cls=cls.backbone_block,
            in_channels=plan_arch["in_channels"],
            start_channels=plan_arch["start_channels"],
            stage_kwargs=None,
            max_channels=max_channels,
            **_kwargs,
        )
        return backbone
