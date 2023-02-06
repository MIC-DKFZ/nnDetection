# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn as nn

"""
Note: register new normalization layers in
nndet.utils.collections to exclude them from weight decay
"""


class GroupNorm(nn.GroupNorm):
    def __init__(
        self,
        num_channels: int,
        num_groups: Optional[int] = None,
        channels_per_group: Optional[int] = None,
        eps: float = 1e-05,
        affine: bool = True,
        **kwargs
    ) -> None:
        """
        PyTorch Group Norm (changed interface, num_channels at first position)

        Args:
            num_channels: number of input channels
            num_groups: number of groups to separate channels. Mutually
                exclusive with `channels_per_group`
            channels_per_group: number of channels per group. Mutually exclusive
                with `num_groups`
            eps: value added to the denom for numerical stability. Defaults to 1e-05.
            affine: Enable learnable per channel affine params. Defaults to True.
        """
        if channels_per_group is not None:
            if num_groups is not None:
                raise ValueError("Can only use `channels_per_group` OR `num_groups` in GroupNorm")
            num_groups = num_channels // channels_per_group

        super().__init__(num_channels=num_channels, num_groups=num_groups, eps=eps, affine=affine, **kwargs)


class LayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        if x.ndim == 4:
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        else:
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x
