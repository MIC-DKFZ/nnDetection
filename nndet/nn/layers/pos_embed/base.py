# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod

import torch


class BasePositionEmbedding(torch.nn.Module):
    def __init__(self, dim: int, num_pos_feats: int) -> None:
        """
        Base class to implement positional embeddings

        Args:
            dim: number of spatial dimensions
            num_pos_feats: number of positional encoding features
        """
        super().__init__()
        self.dim = dim
        if self.dim not in [2, 3]:
            raise ValueError(f"Pos Embedding only supports 2D and 3D, found {self.dim}D")
        self.num_pos_feats = num_pos_feats

    @abstractmethod
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Generate embedding

        Args:
            data: input feature map. [N, C, dims]
            N = batch size, C = number of channels,
            dims = spatial dimensions

        Returns:
            torch.Tensor: spatial embedding [] #TODO: insert here
        """
        raise NotImplementedError
