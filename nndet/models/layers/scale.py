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


class Scale(nn.Module):
    def __init__(self, scale: float = 1.):
        """
        Layer to create a learnable scaling of feature maps

        Args:
            scale: initial value
        """
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inp: input tensor

        Returns:
            Tensor: scaled tensor
        """
        return inp * self.scale

    def extra_repr(self) -> str:
        return f"scale={self.scale.item()}"
