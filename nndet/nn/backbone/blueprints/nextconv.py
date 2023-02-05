# ConvNeXtBlock from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
# SPDX-FileCopyrightText: 2022 Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: MIT

# Changes Licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

try:
    from timm.models.layers import DropPath
except ImportError:
    DropPath = None

from typing import Optional

import torch

from nndet.nn.layers.wrapper import compute_padding_for_kernel, nd_act, nd_conv, nd_norm

if DropPath is not None:
    # TODO: layer norm eps 1e-6 default
    # TODO weight init
    class ConvNeXtBlock(torch.nn.Module):
        def __init__(
            self,
            dim: int,
            in_channels: int,
            kernel_size: int = 7,
            stride: int = 1,
            act_type: str = "GELU",
            act_kwargs: Optional[dict] = None,
            norm_type: str = "Layer",
            norm_kwargs: Optional[dict] = None,
            drop_path: float = 0.0,
            expansion_rate: float = 4.0,
            layer_scale_init_value: Optional[float] = 1e-6,
        ):
            """
            ConvNeXt Block.
            We use channels first implementation here, refer to
            https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
            for more info on this.

            DwConv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv

            Args:
                dim: Number of input channels.
                drop_path: Stochastic depth rate. Default: 0.0
                layer_scale_init_value: Init value for Layer Scale. Default: 1e-6.
            """
            super().__init__()

            _padding = compute_padding_for_kernel(kernel_size)
            self.dwconv = nd_conv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=_padding,
                stride=stride,
            )  # depthwise conv
            self.pwonv1 = nd_conv(
                in_channels=in_channels,
                out_channels=int(expansion_rate * in_channels),
                kernel_size=1,
                padding=0,
                stride=1,
            )
            self.pwonv2 = nd_conv(
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
            self.norm = nd_norm(norm_type=norm_type, dim=dim, **norm_kwargs)

            # activation
            act_kwargs = {} if act_kwargs is None else act_kwargs
            self.act = nd_act(act_type=act_type, dim=dim, **act_kwargs)

            # other
            self.gamma = (
                torch.nn.Parameter(layer_scale_init_value * torch.ones((in_channels)), requires_grad=True)
                if layer_scale_init_value > 0
                else None
            )
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            input = x
            x = self.dwconv(x)
            x = self.norm(x)

            x = self.pwconv1(x)
            x = self.act(x)

            x = self.pwconv2(x)

            if self.gamma is not None:
                x = self.gamma * x
            x = input + self.drop_path(x)
            return x

else:
    ConvNeXtBlock = None
