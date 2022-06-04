# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn


class Scale(nn.Module):
    def __init__(self, scale: float = 1.0):
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
