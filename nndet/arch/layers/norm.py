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

import torch.nn as nn
from typing import Optional

"""
Note: register new normalization layers in
nndet.training.optimizer.NORM_TYPES to exclude them from weight decay
"""


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels: int,
                 num_groups: Optional[int] = None,
                 channels_per_group: Optional[int] = None,
                 eps: float = 1e-05, affine: bool = True, **kwargs) -> None:
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

        super().__init__(num_channels=num_channels,
                         num_groups=num_groups,
                         eps=eps, affine=affine, **kwargs)
