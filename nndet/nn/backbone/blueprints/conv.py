# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Tuple, Union

import torch
from loguru import logger

from nndet.nn.backbone.blueprints.level import (
    BackboneLevel,
    LevelBackbone,
    WrapperBackboneLevel,
)
from nndet.nn.layers.wrapper import compute_padding_for_kernel, nd_pool
from nndet.utils.enums import PoolingMode
from nndet.utils.typing import CONVGEN, ND_INT


class ConvBackbone(LevelBackbone):
    expansion = 2

    def __init__(
        self,
        conv: CONVGEN,
        in_channels: int,
        start_channels: int,
        max_channels: int = 320,
        pooling_mode: Union[PoolingMode, str] = "conv_kernel",
        stem_cfg: Optional[Dict] = None,
        level_cfgs: Optional[List[Dict]] = None,
    ) -> None:
        """
        Backbone with "plain" (conv -> act -> norm) layers

        Args:
            conv: generator to build a conv with optional act, norm etc.
            in_channels: number of input channels (usually number of modalities)
            start_channels: number of channels after initial convolution
            max_channels: maximum number of channels
            pooling_mode: define pooling type. One of 'conv_kernel' |
                'conv_stride' | 'max_kernel' | 'max_stride | 'avg_kernel' |
                'avg_stride'

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

            stem_cfg: configuration parameters of stem. If None, an empty
                dict will be passed. Ignored, since no stem is used here.
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

        """
        self.start_channels = start_channels
        self.max_channels = max_channels
        self.pooling_mode = PoolingMode(pooling_mode.lower())
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            stem_cfg=stem_cfg,
            level_cfgs=level_cfgs,
        )

    @classmethod
    def from_config_plan(
        cls,
        conv: CONVGEN,
        backbone_cfg: dict,
        plan_arch: dict,
    ):
        """
        Instantiate Backbone from given configs

        Args
            conv: conv generator to use for internal convolutions
            backbone_cfg: backbone configuration

                ``"num_conv"`` Union[int, Sequence[int]]
                    [optional] number of convolutions per level. Default 2.

                ``"max_channels"`` int
                    [optional] provide maximum number of channels inside
                    network. If not provided, value from plan will be used.

                ``"pooling_mode"`` str
                    [optional] define a different pooling type. Please refer
                    to the `init` documentation for mor information.
                    Default `conv_kernel`

                ``"backbone_kwargs"`` dict
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
            assert len(num_conv) == num_levels

        if "max_channels" in backbone_cfg:
            max_channels = backbone_cfg.get("max_channels")
            logger.info(f"Found max_channels {max_channels} in backbone config.")
        else:
            max_channels = plan_arch["max_channels"]
        pooling_mode = backbone_cfg.get("pooling_mode", "conv_kernel")

        if "start_channels" in backbone_cfg:
            start_channels = backbone_cfg["start_channels"]
            logger.info(f"Found start_channels {start_channels} in backbone config.")
        else:
            start_channels = plan_arch["start_channels"]

        # build backbone config
        stem_cfg = {}  # no stem to configure
        level_cfgs = []
        for i in range(num_levels):
            _cfg = {
                "kernel": plan_arch["conv_kernels"][i],
                "num_conv": num_conv[i],
                "kwargs": backbone_cfg.get("kwargs", {}),
            }
            if i > 0:
                _cfg["stride"] = plan_arch["strides"][i - 1]
            level_cfgs.append(_cfg)

        backbone = cls(
            conv=conv,
            in_channels=plan_arch["in_channels"],
            start_channels=start_channels,
            pooling_mode=pooling_mode,
            max_channels=max_channels,
            stem_cfg=stem_cfg,
            level_cfgs=level_cfgs,
        )
        return backbone

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
        return self.start_channels, None

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
        _modules = []
        _out_channels = min(
            self.start_channels * (self.expansion**level_idx),
            self.max_channels,
        )
        _kernel = level_cfg["kernel"]
        _padding = compute_padding_for_kernel(_kernel)

        # build first conv
        if level_idx == 0:
            _stride = 1
            _in_channels = self.in_channels
            _modules.append(
                conv(
                    in_channels=_in_channels,
                    out_channels=_out_channels,
                    kernel_size=_kernel,
                    stride=_stride,
                    padding=_padding,
                    **level_cfg.get("kwargs", {}),
                )
            )
        else:
            # deeper levels
            _stride = level_cfg["stride"]
            _in_channels = self.out_channels[-1]
            _modules.append(
                self._build_pooling(
                    conv=conv,
                    in_channels=_in_channels,
                    out_channels=_out_channels,
                    kernel=_kernel,
                    stride=_stride,
                    **level_cfg.get("kwargs", {}),
                )
            )

        # build additional convs
        _num_conv = level_cfg["num_conv"]
        assert _num_conv >= 1
        for _ in range(_num_conv - 1):
            _modules.append(
                conv(
                    in_channels=_out_channels,
                    out_channels=_out_channels,
                    kernel_size=_kernel,
                    stride=1,
                    padding=_padding,
                    **level_cfg.get("kwargs", {}),
                )
            )
        return (
            _out_channels,
            _stride,
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

        if self.pooling_mode.value.startswith("conv"):
            _module = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=_kernel,
                stride=stride,
                padding=_padding,
                **kwargs,
            )
        else:
            _padding_orig_kernel = compute_padding_for_kernel(kernel)
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
