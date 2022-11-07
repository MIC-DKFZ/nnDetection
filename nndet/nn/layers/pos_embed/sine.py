# Modified from https://github.com/facebookresearch/detr
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math

import torch

from nndet.nn.layers.pos_embed.base import BasePositionEmbedding


class PositionEmbeddingSine(BasePositionEmbedding):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        in_channels: int,
        temperature: int = 10000,
        normalize: bool = False,
        scale: bool = None,
        offset: float = 0.0,
    ):
        """
        Implement sine positional embedding for 3D data

        Args:
            in_channels: number of input features
            #TODO: more docs
        """
        super().__init__(
            in_channels=in_channels,
        )
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.offset = offset

    def forward(self, tensor):
        x = tensor
        if tensor.shape[1] % 2 != 0:
            raise Warning("Position Encoding features not divisible by 2")
        if tensor.shape[1] % 6 != 0:
            self.num_pos_feats = tensor.shape[1] // 6 * 2 + 2
        else:
            self.num_pos_feats = tensor.shape[1] // 3
        not_mask = torch.ones(
            (tensor.shape[0], tensor.shape[2], tensor.shape[3], tensor.shape[4]),
            device=x.device,
        )
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        z_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            x_embed = (x_embed + self.offset) / (x_embed[:, -1:, :] + eps) * self.scale
            y_embed = (y_embed + self.offset) / (y_embed[:, :, -1:] + eps) * self.scale
            z_embed = (
                (z_embed + self.offset) / (z_embed[:, :, :, -1:] + eps) * self.scale
            )

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5
        ).flatten(4)
        pos_y = torch.stack(
            (pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5
        ).flatten(4)
        pos_z = torch.stack(
            (pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5
        ).flatten(4)
        if 3 * self.num_pos_feats - tensor.shape[1] == 0:
            pos = torch.cat((pos_x, pos_y, pos_z), dim=4).permute(0, 4, 1, 2, 3)
        elif 3 * self.num_pos_feats - tensor.shape[1] == 2:
            pos = torch.cat((pos_x, pos_y[..., :-1], pos_z[..., :-1]), dim=4).permute(
                0, 4, 1, 2, 3
            )
        elif 3 * self.num_pos_feats - tensor.shape[1] == 4:
            pos = torch.cat(
                (pos_x[..., :-1], pos_y[..., :-1], pos_z[..., :-2]), dim=4
            ).permute(0, 4, 1, 2, 3)
        return pos
