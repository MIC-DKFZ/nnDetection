# ConvNeXtBlock from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
# SPDX-FileCopyrightText: 2022 Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: MIT

# Changes Licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

try:
    from timm.models.layers import DropPath, trunc_normal_
except ImportError:
    DropPath = None

import copy
from functools import reduce
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from loguru import logger

from nndet.nn.backbone.blueprints.conv import ConvBackbone
from nndet.nn.backbone.blueprints.level import (
    BackboneLevel,
    NoOpBackboneLevel,
    WrapperBackboneLevel,
)
from nndet.nn.layers.wrapper import (
    compute_padding_for_kernel,
    nd_act,
    nd_conv,
    nd_norm,
    nd_pool,
)
from nndet.utils.collections import CONV_TYPES
from nndet.utils.info import experimental
from nndet.utils.typing import CONVGEN, ND_INT

if DropPath is not None:

    class ConvNeXtBlock(torch.nn.Module):
        @experimental
        def __init__(
            self,
            dim: int,
            in_channels: int,
            kernel_size: ND_INT,
            stride: ND_INT,
            padding: int,
            act_type: str,
            act_kwargs: Optional[dict],
            norm_type: str,
            norm_kwargs: Optional[dict],
            expansion_rate: float,
            drop_path: float,
            layer_scale_init_value: Optional[float],
            weight_init_fn: Optional[Callable[[torch.nn.Module], None]] = None,
        ):
            """
            ConvNeXt Block.
            We use channels first implementation here, refer to
            https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
            for more info on this.

            DwConv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv

            Args:
                dim: Number of spatial dimensions
                in_channels: number of input channels
                kernel_size: kernel size of depth wise convolution
                stride: stride of convolution
                padding: padding applied to feature map
                act_type: type of activation. Refer to
                    `nndet.nn.layers.wrapper.nd_act` for more information.
                act_kwargs: keyword arguments passed to activation
                    If None and layer norm, `eps=1e-6` is passed as default
                    argument.
                norm_type: type of normalisation. Refer to
                    `nndet.nn.layers.wrapper.nd_norm` for more information.
                norm_kwargs: keyword argument passed to layer norm.
                expansion_rate: expansion ratio used to increase the number
                    of channels
                drop_path: Stochastic depth rate.
                layer_scale_init_value: Init value for Layer Scale.
                    If None and layernorm is selected, the default value of
                    1e-6 is used. Otherwise, if None, no scaling is performed.
                weight_init_fn: function to initialise weights. If None,
                    the default initialisation from the original implementation
                    is used.
            """
            super().__init__()

            # depthwise conv
            self.dwconv = nd_conv(
                dim=dim,
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            )
            self.pwconv1 = nd_conv(
                dim=dim,
                in_channels=in_channels,
                out_channels=int(expansion_rate * in_channels),
                kernel_size=1,
                padding=0,
                stride=1,
            )
            self.pwconv2 = nd_conv(
                dim=dim,
                in_channels=int(expansion_rate * in_channels),
                out_channels=in_channels,
                kernel_size=1,
                padding=0,
                stride=1,
            )
            # norm
            if norm_kwargs is None and norm_type == "Layer":
                norm_kwargs = {"eps": 1e-6}
            elif norm_kwargs is None:
                norm_kwargs = {}
            self.norm = nd_norm(norm_type, dim, in_channels, **norm_kwargs)

            # activation
            act_kwargs = {} if act_kwargs is None else act_kwargs
            self.act = nd_act(act_type=act_type, dim=dim, **act_kwargs)

            # shortcut
            stride_prod = reduce((lambda x, y: x * y), stride) if isinstance(stride, Sequence) else stride
            if stride_prod > 1:
                self.shortcut = torch.nn.Sequential(
                    nd_pool("Avg", dim=dim, kernel_size=stride, stride=stride),
                    nd_conv(
                        dim=dim,
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1,
                    ),
                    nd_norm(norm_type, dim, in_channels, **norm_kwargs),
                )
            else:
                self.shortcut = torch.nn.Identity()

            # other
            if norm_type == "Layer" and layer_scale_init_value is None:
                layer_scale_init_value = 1e-6

            _shape = (1, in_channels, *[1 for _ in range(dim)])
            self.gamma = (
                torch.nn.Parameter(
                    layer_scale_init_value * torch.ones(_shape),
                    requires_grad=True,
                )
                if layer_scale_init_value > 0
                else None
            )

            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()
            self.init_weights(weight_init_fn=weight_init_fn)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forwards tensor through module

            Args:
                x: input tensor [N, C, s_dims] where N is the batch size,
                    C is the number of channels and s_dims are spatial
                    dimensions

            Returns:
                torch.Tensor: output tensor[N, C, s_dims] where N is the
                    batch size, C is the number of channels and s_dims are
                    spatial dimensions
            """
            input = x
            x = self.dwconv(x)
            x = self.norm(x)

            x = self.pwconv1(x)
            x = self.act(x)

            x = self.pwconv2(x)

            if self.gamma is not None:
                x = self.gamma * x
            x = self.shortcut(input) + self.drop_path(x)
            return x

        def init_weights(self, weight_init_fn: Callable[[torch.nn.Module], None]) -> None:
            """
            Initialize weights

            Args:
                weight_init_fn: function to initialise weights. If None,
                    the default initialisation from the original implementation
                    is used.
            """
            if weight_init_fn is None:

                @torch.no_grad()
                def _init_weights(m):
                    if isinstance(m, (*CONV_TYPES, torch.nn.LayerNorm)):
                        trunc_normal_(m.weight, std=0.02)
                        torch.nn.init.constant_(m.bias, 0)

                weight_init_fn = _init_weights
            self.apply(weight_init_fn)

    class ConvNeXtBackbone(ConvBackbone):
        def __init__(
            self,
            conv: CONVGEN,
            in_channels: int,
            start_channels: int,
            max_channels: int = 320,
            stem_cfg: Optional[Dict] = None,
            level_cfgs: Optional[List[Dict]] = None,
        ) -> None:
            """
            Backbone with convnext blocks and conv stem

            Args:
                conv: generator to build a conv with optional act, norm etc.
                    Only used to build the stem and strided conv downsampling.
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

            """
            super().__init__(
                conv=conv,
                in_channels=in_channels,
                start_channels=start_channels,
                max_channels=max_channels,
                stem_cfg=stem_cfg,
                level_cfgs=level_cfgs,
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
            level_num_blocks = level_cfg["num_conv"] // 3
            level_in_channels = self.start_channels if level_idx == 0 else self.out_channels[-1]
            level_out_channels = min(
                self.start_channels * (self.expansion**level_idx),
                self.max_channels,
            )
            # empty level
            if level_num_blocks == 0:
                return (
                    level_in_channels,
                    1,
                    NoOpBackboneLevel(),
                )

            # fill level
            small_level_kernel = level_cfg["small_kernel"]
            small_level_padding = compute_padding_for_kernel(small_level_kernel)
            large_level_kernel = level_cfg["large_kernel"]
            large_level_padding = compute_padding_for_kernel(large_level_kernel)
            level_stride = 1 if level_idx == 0 else level_cfg["stride"]

            if level_cfg["num_conv"] % 3 > 0 and level_idx > 0:
                logger.warning(
                    f"Found {level_cfg['num_conv']} num convs for "
                    f"level {level_idx} in backbone but each "
                    f"block has 3 conv, using {level_num_blocks} res blocks."
                )

            _modules = []
            _modules.append(
                conv(
                    in_channels=level_in_channels,
                    out_channels=level_out_channels,
                    kernel_size=small_level_kernel,
                    stride=level_stride,
                    padding=small_level_padding,
                    add_norm=False,
                    add_act=False,
                )
            )

            for _ in range(level_num_blocks):
                _modules.append(
                    ConvNeXtBlock(
                        dim=conv.dim,
                        in_channels=level_out_channels,
                        # out_channels=level_out_channels,
                        kernel_size=large_level_kernel,
                        stride=1,
                        padding=large_level_padding,
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

                    ``'large_kernel_size'`` int
                        [optional] provide large kernel size. Default 7.

                    ``'act_type'`` str
                        [optional] type of activation. Refer to
                        `nndet.nn.layers.wrapper.nd_act` for more information.
                        Default `GELU`.

                    ``'act_kwargs'`` dict
                        [optional] keyword arguments passed to activation. If
                        None and layer norm, `eps=1e-6` is passed as default
                        argument. Default: None.

                    ``'norm_type'`` str
                        [optional] type of normalisation. Refer to
                        `nndet.nn.layers.wrapper.nd_norm` for more information.
                        Default `Layer`.

                    ``'norm_kwargs'`` dict
                        [optional] keyword argument passed to layer norm.
                        Default: None

                    ``'expansion_rate'`` int
                        [optional] expansion ratio used to increase the number
                        of channels. Default 4.

                    ``'drop_path'`` float
                        [optional] Stochastic depth rate. Default: 0.0

                    ``'layer_scale_init_value'`` float
                        [optional] Init value for Layer Scale. If None and
                        layernorm is selected, the default value of 1e-6 is
                        used. Otherwise, if None, no scaling is performed.
                        Default None.

                    ``'p0_block'`` bool
                        [optional] Add block to P0 in additional to
                        conv stem. Default: False

                    ``"num_conv"`` Union[int, Sequence[int]]
                        [optional] number of convolutions per level. Default 3.

                    ``'num_conv_stem'`` int
                        [optional] number of convolutions in stem. Default to
                        provided 2.

                    ``"max_channels"`` int
                        [optional] provide maximum number of channels inside
                        network. If not provided, value from plan will be used.

                    ``"stem_kwargs"`` dict
                        [optional] keyword arguments passed to stem

                    ``"kwargs"`` dict
                        [optional] keyword arguments passed to every level of
                        the backbone

                plan_arch: arguments provided plan

                    ``"conv_kernels"`` List[ND_INT]
                        kernel size of each level [N]. Expected to be in
                        standard nnDetection format where the large
                        kernel size is indicated as 3.

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

                Notes:
                    downsampling convolutions do not count towards the
                    num_conv count.
            """
            logger.info(f"Building:: backbone {cls.__name__}: {backbone_cfg} ")
            # parse config and plan
            num_levels = len(plan_arch["conv_kernels"])
            num_conv = backbone_cfg.get("num_conv", 3)
            if isinstance(num_conv, int):
                num_conv = [num_conv] * num_levels
            else:
                if len(num_conv) < num_levels:
                    logger.info(
                        f"Found {num_conv} convolutions which is less " "than num levels, filling up with last number."
                    )
                    num_conv = num_conv + [num_conv[-1]] * (num_levels - len(num_conv))
                assert len(num_conv) == num_levels

            num_conv_stem = backbone_cfg.get("num_conv_stem", 2)

            if "max_channels" in backbone_cfg:
                max_channels = backbone_cfg.get("max_channels")
                logger.info(f"Found max_channels {max_channels} in backbone config.")
            else:
                max_channels = plan_arch["max_channels"]

            if "start_channels" in backbone_cfg:
                start_channels = backbone_cfg["start_channels"]
                logger.info(f"Found start_channels {start_channels} in backbone config.")
            else:
                start_channels = plan_arch["start_channels"]

            # retrieve default values
            p0_block = backbone_cfg.get("p0_block", False)
            large_kernel_size = backbone_cfg.get("large_kernel_size", 7)
            act_type = backbone_cfg.get("act_type", "GELU")
            act_kwargs = backbone_cfg.get("act_kwargs", None)
            norm_type = backbone_cfg.get("norm_type", "Layer")
            norm_kwargs = backbone_cfg.get("norm_kwargs", None)
            expansion_rate = backbone_cfg.get("expansion_rate", 4.0)
            drop_path = backbone_cfg.get("drop_path", 0.0)
            layer_scale_init_value = backbone_cfg.get("layer_scale_init_value", None)

            _defaults = {
                "act_type": act_type,
                "act_kwargs": act_kwargs,
                "norm_type": norm_type,
                "norm_kwargs": norm_kwargs,
                "expansion_rate": expansion_rate,
                "drop_path": drop_path,
                "layer_scale_init_value": layer_scale_init_value,
            }

            # build configs
            level_cfgs = []
            for i in range(num_levels):
                if i == 0 and not p0_block:
                    _cfg = {"num_conv": 0}
                else:
                    small_kernel = plan_arch["conv_kernels"][i]
                    large_kernel = cls.replace_kernel_size(small_kernel, large_kernel_size)
                    kwargs = copy.deepcopy(_defaults)
                    kwargs.update(backbone_cfg.get("kwargs", {}))
                    _cfg = {
                        "small_kernel": small_kernel,
                        "large_kernel": large_kernel,
                        "num_conv": num_conv[i],
                        "kwargs": kwargs,
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
            )
            return backbone

        @staticmethod
        def replace_kernel_size(kernel: ND_INT, kernel_size: int) -> ND_INT:
            """
            Replace kernel size 3 with larger kernel sizes

            Args:
                kernel: kernel as integer or tuple of int's. The small kernel
                    size should be 3.
                kernel_size: All small kernels are replaced with this new
                    kernel size.

            Returns:
                ND_INT: Updated kernel size
            """
            if isinstance(kernel, (float, int)):
                if kernel > 1:
                    kernel = kernel_size
            else:
                max_k = max(kernel)
                if max_k > 1:
                    kernel = [kernel_size if k == max_k else k for k in kernel]
            return kernel

else:
    ConvNeXtBlock = None
