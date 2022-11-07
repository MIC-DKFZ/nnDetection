# Modified to support 3D data
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

# Original code from https://github.com/facebookresearch/detr licensed under:
# SPDX-FileCopyrightText: 2020 Facebook
# SPDX-License-Identifier: Apache-2.0

import torch

from nndet.nn.layers.pos_embed.base import BasePositionEmbedding


class PositionEmbeddingLearned(BasePositionEmbedding):
    def __init__(self, dim: int, num_pos_feats: int, num_embeddings: int = 64):
        """
        Positional embedding learned

        Args:
            dim: number of spatial dimensions
            num_pos_feats: number of positional encoding features (d in formula)
            num_embeddings: #TODO
        """
        super().__init__(dim=dim, num_pos_feats=num_pos_feats)

        if self.dim == 3 and self.num_pos_feats % 3 != 0:
            raise ValueError("Sine encoding can only be used if num_pos_feats is divisible by 3 (in 3D)")
        if self.dim == 2 and self.num_pos_feats % 3 != 0:
            raise ValueError("Sine encoding can only be used if num_pos_feats is divisible by 2 (in 2D)")

        if self.dim == 2:
            _num_pos_feats = num_pos_feats // 2
        else:
            _num_pos_feats = num_pos_feats // 3

        self.ax0_embed = torch.nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=_num_pos_feats,
        )
        self.ax1_embed = torch.nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=_num_pos_feats,
        )
        if self.dim == 3:
            self.ax2_embed = torch.nn.Embedding(
                num_embeddings=num_embeddings,
                embedding_dim=_num_pos_feats,
            )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.ax0_embed.weight)
        torch.nn.init.uniform_(self.ax1_embed.weight)
        if self.dim == 3:
            torch.nn.init.uniform_(self.ax2_embed.weight)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute positional embedding

        Args:
            data: input feature map to compute embedding for

        Raises:
            Warning: _description_ # TODO

        Returns:
            torch.Tensor: computed embedding [N, num_pos_feats, dims] where
                N = batch size, dims = spatial dimensions
        """
        if data.ndim == 4:  # 2D
            num_batch, _, ax0, ax1 = tuple(data.shape)
        else:  # 3D
            num_batch, _, ax0, ax1, ax2 = tuple(data.shape)

        i = torch.arange(ax0, device=data.device)
        j = torch.arange(ax1, device=data.device)

        ax0_emb = self.ax0_embed(i)  # [ax0, num_pos_feats]
        ax1_emb = self.ax1_embed(j)  # [ax1, num_pos_feats]

        if data.ndim == 4:
            pos = (
                torch.cat(
                    [
                        ax0_emb.unsqueeze(1).repeat(1, ax1, 1),  # [ax0, ax1, num_pos_feats]
                        ax1_emb.unsqueeze(0).repeat(ax0, 1, 1),  # [ax0, ax1, num_pos_feats]
                    ],
                    dim=-1,
                )
                .permute(2, 0, 1)
                .unsqueeze(0)
                .repeat(num_batch, 1, 1, 1)
            )
        else:
            k = torch.arange(ax2, device=data.device)
            ax2_emb = self.ax2_embed(k)  # [ax2, num_pos_feats]

            pos = (
                torch.cat(
                    [
                        ax0_emb.unsqueeze(1).unsqueeze(2).repeat(1, ax1, ax2, 1),  # [ax0, ax1, ax2, num_pos_feats]
                        ax1_emb.unsqueeze(0).unsqueeze(2).repeat(ax0, 1, ax2, 1),  # [ax0, ax1, ax2, num_pos_feats]
                        ax2_emb.unsqueeze(0).unsqueeze(1).repeat(ax0, ax1, 1, 1),  # [ax0, ax1, ax2, num_pos_feats]
                    ],
                    dim=-1,
                )
                .permute(3, 0, 1, 2)
                .unsqueeze(0)
                .repeat(num_batch, 1, 1, 1, 1)
            )

        return pos
