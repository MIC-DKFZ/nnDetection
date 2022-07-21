# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

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
        self.scale = nn.Parameter(
            torch.tensor(scale, dtype=torch.float),
            requires_grad=True,
        )

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


class ScalePerDim(nn.Module):
    def __init__(self, scale: Sequence[float] = [1.0, 1.0, 1.0]):
        """
        Layer to create a learnable scaling per spatial dimension of
        feature maps

        Args:
            scale: initial value
        """
        super().__init__()
        self.dim: int = len(scale)
        if len(scale) not in [2, 3]:
            raise ValueError(f"Scale need to be 2 or 3 dimensional, found {scale}")
        self.scale = nn.Parameter(
            torch.tensor(scale, dtype=torch.float),
            requires_grad=True,
        )

        # _scales = [torch.tensor(s, dtype=torch.float) for s in scale]
        # if len(scale) == 2:
        #     scale_tensor = torch.stack([_scales[0], _scales[1]])
        # elif len(scale) == 3:
        #     scale_tensor = torch.stack([_scales[0], _scales[1], _scales[2]])
        # else:
        #     raise ValueError(f"Scale need to be 2 or 3 dimensional, found {scale}")
        # self.scale = nn.Parameter(scale_tensor[None, None], requires_grad=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inp: input tensor

        Returns:
            Tensor: scaled tensor
        """
        if self.dim == 3:
            _scale = torch.stack(
                [
                    self.scale[0],
                    self.scale[1],
                    self.scale[0],
                    self.scale[1],
                    self.scale[2],
                    self.scale[2],
                ]
            )[None, None]
        else:
            _scale = torch.stack(
                [
                    self.scale[0],
                    self.scale[1],
                    self.scale[0],
                    self.scale[1],
                ],
            )[None, None]
        return inp * _scale

    def extra_repr(self) -> str:
        return f"scale={self.scale}"
