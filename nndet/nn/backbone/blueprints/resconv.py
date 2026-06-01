# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from functools import reduce
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from loguru import logger

from nndet.nn.backbone.blueprints.conv import ConvBackbone
from nndet.nn.backbone.blueprints.level import (
    BackboneLevel,
    NoOpBackboneLevel,
    WrapperBackboneLevel,
)
from nndet.nn.layers.wrapper import compute_padding_for_kernel, nd_pool
from nndet.utils.enums import PoolingMode
from nndet.utils.typing import CONVGEN, ND_INT


class ResPlain(nn.Module):
    def __init__(
        self,
        conv: CONVGEN,
        in_channels: int,
        out_channels: int,
        kernel_size: ND_INT,
        stride: ND_INT,
        padding: ND_INT,
        attention: Optional[nn.Module] = None,
    ):
        """
        Build a plain residual block
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

        Warning:
            The second convolotion always uses a ReLU activation independent of
            the selected conf activation.

        """
        super().__init__()
        self.conv1 = conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.conv2 = conv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            add_act=False,
        )
        self.relu = nn.ReLU(inplace=True)

        stride_prod = reduce((lambda x, y: x * y), stride) if isinstance(stride, Sequence) else stride
        if stride_prod > 1:
            self.shortcut = nn.Sequential(
                nd_pool("Avg", dim=conv.dim, kernel_size=stride, stride=stride),
                conv(in_channels, out_channels, kernel_size=1, add_act=False),
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

        if self.attention is not None:
            out = self.attention(out)
        if self.shortcut is not None:
            residual = self.shortcut(x)

        out += residual
        out = self.relu(out)
        return out

    def init_weights(self) -> None:
        try:
            torch.nn.init.zeros_(self.conv2.norm.weight)
        except BaseException:
            logger.info(f"Zero init of second conv norm layer {self.conv2.norm} failed")


class ResConvBackbone(ConvBackbone):
    def __init__(
        self,
        conv: CONVGEN,
        in_channels: int,
        start_channels: int,
        max_channels: int = 320,
        stem_cfg: Optional[Dict] = None,
        level_cfgs: Optional[List[Dict]] = None,
        pooling_mode: Union[PoolingMode, str] = "block",
    ) -> None:
        """
        Backbone with plain residual blocks abd conv stem

        Args:
            conv: generator to build a conv with optional act, norm etc.
            in_channels: number of input channels (usually number of modalities)
            start_channels: number of channels after initial convolution
            max_channels: maximum number of channels

            stem_cfg: configuration parameters of stem. If None, an empty
                dict will be passed. Ignored, since no stem is used here.

                ``'kernel'``
                    kernel size for each level

                ``'num_conv'``
                    number of convs per level

                ``'kwargs'``
                    keyword arguments passed to conv in level

            level_cfgs: configuration for each level. If None, an empty
                dict will be passed.

                ``'kernel'``
                    kernel size for each level

                ``'stride'``
                    stride for levels starting from 1

                ``'num_conv'``
                    number of convs per level

                ``'kwargs'``
                    keyword arguments passed to conv in level

            pooling_mode: define pooling type. One of 'block' |
                'conv_kernel' | 'conv_stride' | 'max_kernel' | 'max_stride |
                'avg_kernel' | 'avg_stride'

                ``'block'``
                    residual block for downsampling

                ``'conv_kernel'``
                    uses strided convolutions with same kernel
                    size as respective layer for pooling.

                ``'conv_stride'``
                    uses strided convolutions with kernel size
                    matching the stride (non overlapping) for pooling

                ``'max_kernel'``
                    max pooling where pooling kernal equals conv
                    kernel of respective layer

                ``'max_stride'``
                    max pooling with kernel size matching the
                    stride (non overlapping)

                ``'avg_kernel'``
                    same as max with average pooling

                ``'avg_stride'``
                    same as max with average pooling

        """
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            start_channels=start_channels,
            max_channels=max_channels,
            stem_cfg=stem_cfg,
            level_cfgs=level_cfgs,
            pooling_mode=pooling_mode,
        )

    def _build_stem(
        self,
        conv: CONVGEN,
        stem_cfg: Dict,
    ) -> Tuple[int, None]:
        """
        ConvBackbone does not use a special stem.

        Args:
            in_channels: number of input channels (usually number of modalities)
            stem_config: configuration of stem

        Returns:
            Tuple[int, torch.nn.Module]: The first output is the number of
                output channels of the stem. The second output is the
                stem itself.
        """
        _in_channels = self.in_channels
        _out_channels = min(self.start_channels, self.max_channels)
        _kernel = stem_cfg["kernel"]
        _padding = compute_padding_for_kernel(_kernel)
        _stride = 1
        _num_conv = stem_cfg["num_conv"]

        _modules = []
        _modules.append(
            conv(
                in_channels=_in_channels,
                out_channels=_out_channels,
                kernel_size=_kernel,
                stride=_stride,
                padding=_padding,
                **stem_cfg.get("kwargs", {}),
            )
        )
        assert _num_conv >= 1
        for _ in range(_num_conv - 1):
            _modules.append(
                conv(
                    in_channels=_out_channels,
                    out_channels=_out_channels,
                    kernel_size=_kernel,
                    stride=1,
                    padding=_padding,
                    **stem_cfg.get("kwargs", {}),
                )
            )
        return self.start_channels, torch.nn.Sequential(*_modules)

    def _build_level(
        self,
        conv: CONVGEN,
        level_idx: int,
        level_cfg: Dict,
    ) -> Tuple[int, ND_INT, BackboneLevel]:
        """
        Build one backbone level. Each level returns the features to propagate
        deeper through the network and the features which should be returned
        by the backbone.

        Args:
            level_idx: index of level
            level_cfg: configuration of level

        Returns:
            int: number of output channels
            ND_INT: (relative) stride of level
            BackboneLevel: constructed level
        """
        assert self.pooling_mode == PoolingMode.BLOCK, "Only block supported for downsampling"

        level_num_blocks = level_cfg["num_conv"] // 2
        level_in_channels = self.start_channels if level_idx == 0 else self.out_channels[-1]
        if level_num_blocks == 0:
            return (
                level_in_channels,
                1,
                NoOpBackboneLevel(),
            )

        level_out_channels = min(
            self.start_channels * (self.expansion**level_idx),
            self.max_channels,
        )
        level_kernel = level_cfg["kernel"]
        level_padding = compute_padding_for_kernel(level_kernel)
        level_stride = 1 if level_idx == 0 else level_cfg["stride"]

        if level_cfg["num_conv"] % 2 > 0 and level_idx > 0:
            logger.warning(
                f"Found {level_cfg['num_conv']} num convs for "
                f"level {level_idx} in backbone but each residual "
                f"block has 2 conv, using {level_num_blocks} res blocks."
            )

        _modules = []
        _modules.append(
            ResPlain(
                conv=conv,
                in_channels=level_in_channels,
                out_channels=level_out_channels,
                kernel_size=level_kernel,
                stride=level_stride,
                padding=level_padding,
                attention=None,
                **level_cfg.get("kwargs", {}),
            )
        )

        for _ in range(level_num_blocks - 1):
            _modules.append(
                ResPlain(
                    conv=conv,
                    in_channels=level_out_channels,
                    out_channels=level_out_channels,
                    kernel_size=level_kernel,
                    stride=1,
                    padding=level_padding,
                    attention=None,
                    **level_cfg.get("kwargs", {}),
                )
            )
        return (
            level_out_channels,
            level_stride,
            WrapperBackboneLevel(torch.nn.Sequential(*_modules)),
        )

    @classmethod
    def from_config_plan(
        cls,
        conv: CONVGEN,
        backbone_cfg: dict,
        plan_arch: dict,
    ):
        """
        Instantiate Backbone from given configs.

        Args
            conv: conv generator to use for internal convolutions
            backbone_cfg: backbone configuration

                ``'res_p0'`` bool
                    [optional] Add residual block to P0 in additional to
                    conv stem. Default: False

                ``"num_conv"`` Union[int, Sequence[int]]
                    [optional] number of convolutions per level. Default 2.

                ``'num_conv_stem'`` int
                    [optional] number of convolutions in stem. Default to
                    provided `num_conv`.

                ``"max_channels"`` int
                    [optional] provide maximum number of channels inside
                    network. If not provided, value from plan will be used.

                ``"stem_kwargs"`` dict
                    [optional] keyword arguments passed to stem

                ``"pooling_mode"`` str
                    [optional] define a different pooling type. Please refer
                    to the `init` documentation for mor information.
                    Default `block`

                ``"kwargs"`` dict
                    [optional] keyword arguments passed to every level of the
                    backbone

            plan_arch: arguments provided plan

                ``"conv_kernels"`` List[ND_INT]
                    kernel size of each level [N]

                ``"strides"`` List[ND_INT]
                    stride for each level [N - 1]

                ``"in_channels"`` List[ND_INT]
                    Number of input channels, usually equal to number of
                    modalities.

                ``"start_channels"`` int
                    number of start channels, i.e. number of channels after
                    first conv. Can be overwritten via config.

                ``"max_channels"`` int
                    maximum number of channels inside model, usually 320.
                    Can be overwritten via config.

        """
        logger.info(f"Building:: backbone {cls.__name__}: {backbone_cfg} ")
        # parse config and plan
        num_levels = len(plan_arch["conv_kernels"])
        num_conv = backbone_cfg.get("num_conv", 2)
        if isinstance(num_conv, int):
            num_conv = [num_conv] * num_levels
        else:
            if len(num_conv) < num_levels:
                logger.info(
                    f"Found {num_conv} convolutions which is less " "than num levels, filling up with last number."
                )
                num_conv = num_conv + [num_conv[-1]] * (num_levels - len(num_conv))
            assert len(num_conv) == num_levels

        num_conv_stem = backbone_cfg.get("num_conv_stem", num_conv[0])

        if "max_channels" in backbone_cfg:
            max_channels = backbone_cfg.get("max_channels")
            logger.info(f"Found max_channels {max_channels} in backbone config.")
        else:
            max_channels = plan_arch["max_channels"]
        pooling_mode = backbone_cfg.get("pooling_mode", "block")

        if "start_channels" in backbone_cfg:
            start_channels = backbone_cfg["start_channels"]
            logger.info(f"Found start_channels {start_channels} in backbone config.")
        else:
            start_channels = plan_arch["start_channels"]

        # build backbone config
        res_p0 = backbone_cfg.get("res_p0", False)
        level_cfgs = []
        for i in range(num_levels):
            if i == 0 and not res_p0:
                _cfg = {"num_conv": 0}
            else:
                _cfg = {
                    "kernel": plan_arch["conv_kernels"][i],
                    "num_conv": num_conv[i],
                    "kwargs": backbone_cfg.get("kwargs", {}),
                }
            if i > 0:
                _cfg["stride"] = plan_arch["strides"][i - 1]
            level_cfgs.append(_cfg)

        stem_cfg = {
            "kernel": plan_arch["conv_kernels"][0],
            "num_conv": num_conv_stem,
            "kwargs": backbone_cfg.get("stem_kwargs", {}),
        }

        backbone = cls(
            conv=conv,
            in_channels=plan_arch["in_channels"],
            start_channels=start_channels,
            max_channels=max_channels,
            stem_cfg=stem_cfg,
            level_cfgs=level_cfgs,
            pooling_mode=pooling_mode,
        )
        return backbone


class ResConvWithPoolBackbone(ResConvBackbone):
    def _build_level(
        self,
        conv: CONVGEN,
        level_idx: int,
        level_cfg: Dict,
    ) -> Tuple[int, ND_INT, BackboneLevel]:
        """
        Build one backbone level. Each level returns the features to propagate
        deeper through the network and the features which should be returned
        by the backbone. In contrast to `ResConvBackbone` this implements
        multiple downsampling methods and the pooling layer does  *not*
        include the downsampling layers.

        Args:
            level_idx: index of level
            level_cfg: configuration of level

        Returns:
            int: number of output channels
            ND_INT: (relative) stride of level
            BackboneLevel: constructed level
        """
        level_num_blocks = level_cfg["num_conv"] // 2
        level_in_channels = self.start_channels if level_idx == 0 else self.out_channels[-1]
        if level_num_blocks == 0:
            return (
                level_in_channels,
                1,
                NoOpBackboneLevel(),
            )

        level_out_channels = min(
            self.start_channels * (self.expansion**level_idx),
            self.max_channels,
        )
        level_kernel = level_cfg["kernel"]
        level_padding = compute_padding_for_kernel(level_kernel)
        level_stride = 1 if level_idx == 0 else level_cfg["stride"]

        if level_cfg["num_conv"] % 2 > 0 and level_idx > 0:
            logger.warning(
                f"Found {level_cfg['num_conv']} num convs for "
                f"level {level_idx} in backbone but each residual "
                f"block has 2 conv, using {level_num_blocks} res blocks."
            )

        _modules = []
        _modules.append(
            self._build_pooling(
                conv=conv,
                in_channels=level_in_channels,
                out_channels=level_out_channels,
                kernel=level_kernel,
                stride=level_stride,
                **level_cfg.get("kwargs", {}),
            )
        )

        for _ in range(level_num_blocks):
            _modules.append(
                ResPlain(
                    conv=conv,
                    in_channels=level_out_channels,
                    out_channels=level_out_channels,
                    kernel_size=level_kernel,
                    stride=1,
                    padding=level_padding,
                    attention=None,
                    **level_cfg.get("kwargs", {}),
                )
            )
        return (
            level_out_channels,
            level_stride,
            WrapperBackboneLevel(torch.nn.Sequential(*_modules)),
        )

    def _build_pooling(
        self,
        conv: CONVGEN,
        in_channels: int,
        out_channels: int,
        kernel: ND_INT,
        stride: ND_INT,
        **kwargs,
    ) -> torch.nn.Module:
        """
        Build pooling layer of respective level

        Args:
            conv: generator to build a conv with optional act, norm etc.
            in_channels: number of input channels (usually number of modalities)
            out_channels: number of output channels
            kernel: kernel size
            stride: stride

        Returns:
            torch.nn.Module: created pooling module
        """
        # if non overlapping stride is needed -> set kernel to stride
        if self.pooling_mode.value.endswith("stride"):
            _kernel = stride
        else:
            _kernel = kernel
        _padding = compute_padding_for_kernel(_kernel)
        _padding_orig_kernel = compute_padding_for_kernel(kernel)

        if self.pooling_mode == PoolingMode.BLOCK:
            _module = ResPlain(
                conv=conv,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=_padding_orig_kernel,
                attention=None,
                **kwargs,
            )
        elif self.pooling_mode.value.startswith("conv"):
            _module = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=_kernel,
                stride=stride,
                padding=_padding,
                **kwargs,
            )
        else:
            _pool_type = self.pooling_mode.value.split("_")[0].capitalize()
            _module = torch.nn.Sequential(
                nd_pool(
                    pooling_type=_pool_type,
                    dim=conv.dim,
                    kernel_size=_kernel,
                    stride=stride,
                    padding=_padding,
                ),
                conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    stride=1,
                    padding=_padding_orig_kernel,
                    **kwargs,
                ),
            )
        return _module
