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

"""
Don't use these. Next nnDetection Version will introduce better/fixed implementations.
"""

import torch
import torch.nn as nn


from nndet.arch.conv import nd_pool, nd_conv


class SELayer(nn.Module):
    def __init__(self,
                 dim: int,
                 in_channels: int,
                 reduction: int = 16,
                 ):
        """
        Squeeze and Excitation Layer
        https://arxiv.org/abs/1709.01507

        Args
            dim: number of spatial dimensions
            in_channels: number of input channels
            reduction: channel reduction for internal computations
        """
        super(SELayer, self).__init__()
        self.pool = nd_pool("AdaptiveAvg", dim, 1)
        self.fc = nn.Sequential(
            nd_conv(dim, in_channels, in_channels // reduction,
                    kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nd_conv(dim, in_channels // reduction, in_channels,
                    kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)
        y = self.fc(y)
        return x * y
