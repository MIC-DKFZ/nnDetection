# Modified to support 3D data
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

# Original code from https://github.com/facebookresearch/detr licensed under:
# SPDX-FileCopyrightText: 2020 Facebook
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from nndet.nn.layers.pos_embed.base import BasePositionEmbedding


class PositionEmbeddingSine(BasePositionEmbedding):
    def __init__(
        self,
        dim: int,
        num_pos_feats: int,
        temperature: int = 10000,
        normalize: bool = False,
        scale: bool = None,
        offset: float = 0.0,
    ):
        """
        Implement sine positional embedding for 2/3D data

        It computed the encoding along each dimension separately and
        concatenates them at the end. Each dimension follows this rule:

            p_i = sin(w_k * t) if i = 2k cos(w_k * t) if i = 2k + 1
            w_k = 1 / temperature ** (2 * k / d)

        Args:
            dim: number of spatial dimensions
            num_pos_feats: number of positional encoding features (d in formula)
            temperature: term in denominator to compute position
            noramlize: normalize t to the [0, 1] range
            scale: scale t to different range, only applicable if normalize is
                set to `True`
            offset: add offset to t, only applicable if normalize is set to
                `True`
        """
        super().__init__(
            dim=dim,
            num_pos_feats=num_pos_feats,
        )

        if self.dim == 3 and self.num_pos_feats % 6 != 0:
            raise ValueError("Sine encoding can only be used if num_pos_feats is divisible by 3 (in 3D)")
        if self.dim == 2 and self.num_pos_feats % 4 != 0:
            raise ValueError("Sine encoding can only be used if num_pos_feats is divisible by 2 (in 2D)")

        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if not math.isclose(offset, 0) and normalize is False:
            raise ValueError("normalize should be True if offset is passed")

        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.offset = offset
        self.eps = 1e-6

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute positional embedding

        Args:
            data: input feature map to compute embedding for

        Returns:
            torch.Tensor: computed embedding [N, num_pos_feats, dims] where
                N = batch size, dims = spatial dimensions
        """
        if self.dim == 3:
            _num_pos_feats = self.num_pos_feats // 3
        else:
            _num_pos_feats = self.num_pos_feats // 2

        if data.ndim == 4:  # 2D
            stack_dim = 4
            num_batch, _, ax0, ax1 = tuple(data.shape)
            not_mask = torch.ones((num_batch, ax0, ax1), device=data.device)
        else:  # 3D
            stack_dim = 5
            num_batch, _, ax0, ax1, ax2 = tuple(data.shape)
            not_mask = torch.ones((num_batch, ax0, ax1, ax2), device=data.device)

        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        if data.ndim == 5:
            z_embed = not_mask.cumsum(3, dtype=torch.float32)
            if self.normalize:
                z_embed = (z_embed + self.offset) / (z_embed[..., -1:] + self.eps) * self.scale

        if self.normalize:
            x_embed = (x_embed + self.offset) / (x_embed[:, -1:, :] + self.eps) * self.scale
            y_embed = (y_embed + self.offset) / (y_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = torch.arange(_num_pos_feats, dtype=torch.float32, device=data.device)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / _num_pos_feats
        )  # 2 * (dim_t // 2) / num_feats is used to create an alternating sequence of 0, 1

        pos_x = x_embed[..., None] / dim_t  # [batch, ax0, ax1(, ax2), _num_pos_feats]
        pos_y = y_embed[..., None] / dim_t  # [batch, ax0, ax1(, ax2), _num_pos_feats]

        # [batch, ax0, ax1(, ax2), 2]
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=stack_dim).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=stack_dim).flatten(-2)

        if data.ndim == 5:
            pos_z = z_embed[..., None] / dim_t  # [batch, ax0, ax1(, ax2), _num_pos_feats]
            pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=stack_dim).flatten(-2)
            pos = torch.cat((pos_x, pos_y, pos_z), dim=4).permute(0, 4, 1, 2, 3)
        else:
            pos = torch.cat((pos_x, pos_y), dim=3).permute(0, 3, 1, 2)
        return pos
