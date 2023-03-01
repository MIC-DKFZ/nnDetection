# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from detrex licensed under
# SPDX-FileCopyrightText: 2022, The IDEA Authors
# SPDX-License-Identifier: Apache-2.0

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from nndet.nn.transformer.attention.conditional_attention import (
    ConditionalCrossAttention,
    ConditionalSelfAttention,
)
from nndet.nn.transformer.layers.abstract import AbstractTransformerDecoder
from nndet.nn.transformer.layers.base_layer import (
    BaseTransformerLayer,
    TransformerLayerSequence,
)
from nndet.utils.fully_connected import FCN, SimpleFCN


def gen_sine_embed_for_position(
    pos_tensor: torch.Tensor,
    num_pos_feats: int,
    temperature: int = 10000,
) -> torch.Tensor:
    """
    2D or 3D Positional Encoding to encode given positions (different to the
    normal position encoding which computes position based on pixels)
    Args:
        pos_tensor: tensor of shape (bs, num_pos, dim)
        num_pos_feats: number of out features (output dimension)
        temperature: temperature of the position encoding
    Returns:
        Tensor: tensor containing position embedding
    """
    dim = pos_tensor.shape[2]
    assert dim in [2, 3]

    scale = 2 * math.pi
    if num_pos_feats % dim == 0:
        feats = num_pos_feats // dim
    else:
        feats = num_pos_feats // dim + 1
    dim_t = torch.arange(feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / feats)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

    # Handle 3D Case
    if dim == 3:
        z_embed = pos_tensor[:, :, 2] * scale
        pos_z = z_embed[:, :, None] / dim_t
        pos_z = torch.stack((pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=3).flatten(2)
        # If num_pos_feats is not divisible by 3 we have to
        if num_pos_feats % dim == 0:
            return torch.cat((pos_x, pos_y, pos_z), dim=2)
        elif num_pos_feats % dim == 1:
            return torch.cat((pos_x, pos_y, pos_z[:, :, :-1]), dim=2)
        else:
            return torch.cat((pos_x, pos_y[:, :, :-1], pos_z[:, :, :-1]), dim=2)
    # 2D Case
    if num_pos_feats % dim == 0:
        return torch.cat((pos_x, pos_y), dim=2)
    return torch.cat((pos_x, pos_y[:, :, :-1]), dim=2)


class ConditionalDETRTransformerDecoder(AbstractTransformerDecoder):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        feedforward_dim: int = 2048,
        ffn_dropout: float = 0.1,
        activation: nn.Module = nn.ReLU(),
        post_norm: bool = True,
        return_intermediate: bool = True,
        dim: int = 3,
        batch_first: bool = False,
    ):
        """
        Transformer Decoder for Conditional DETR

        Args:
            embed_dim: embed dimension (hidden dimension) of the transformer
                decoder
            num_heads: number of attention heads
            num_layers: number of decoder layers
            attn_dropout: dropout in the attention modules
            proj_dropout: dropout of the final linear projection after attention
            feedforward_dim: hidden dimension of the feed forward network in the
                transformer layer
            ffn_dropout: dropout of the feed forward network
            activation: activation of the feed forward network
            post_norm: apply an additional layer norm to all outputs
            return_intermediate: return the outputs of all
            dim: dimension of the input, has to be 2 or 3
            batch_first: use batch first computations in the transformer
        """
        super().__init__()
        self.layer_sequence = TransformerLayerSequence(
            transformer_layers=BaseTransformerLayer(
                attn=[
                    ConditionalSelfAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop_value=attn_dropout,
                        proj_drop_value=proj_dropout,
                        batch_first=batch_first,
                    ),
                    ConditionalCrossAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop_value=attn_dropout,
                        proj_drop_value=proj_dropout,
                        batch_first=batch_first,
                    ),
                ],
                ffn=FCN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    ffn_drop=ffn_dropout,
                    activation=activation,
                ),
                norm=nn.LayerNorm(
                    normalized_shape=embed_dim,
                ),
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.return_intermediate = return_intermediate
        self.embed_dim = embed_dim
        self.query_scale = SimpleFCN(self.embed_dim, self.embed_dim, self.embed_dim, 2)
        self.ref_point_head = SimpleFCN(self.embed_dim, self.embed_dim, dim, 2)
        self.dim = dim
        self.bbox_embed = None

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

        for idx in range(num_layers - 1):
            self.layer_sequence.layers[idx + 1].attentions[1].query_pos_proj = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_masks: Optional[List[torch.Tensor]] = None,
        query_key_padding_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute a sequence of output box embeddings given object queries and
            features.

        Args:
            query: Object queries (num_queries, bs, C)
            key: features from the transformer encoder used as keys in
                cross-attention
            value: features from the transformer encoder used as values in
                cross-attention
            query_pos: (Optional) position embedding for the given query
                shape (sequence_length, bs, C)
            key_pos: (Optional) position embedding for the given key
            attn_masks: (Optional) mask for the attention layer
            query_key_padding_mask: (Optional) query key padding mask for
                attention
            key_padding_mask: (Optional) key padding mask for attention
            **kwargs:

        Returns:
            Tensor: Sequence of output embeddings, either of the last layer if
                return_intermediate is false  or of all layers with shape
                ((num_decoder_layers), num_queries, bs, C)
        """

        intermediate = []
        reference_points_before_sigmoid = self.ref_point_head(query_pos)  # [num_queries, batch_size, dim]
        reference_points = reference_points_before_sigmoid.sigmoid().transpose(0, 1)

        for idx, layer in enumerate(self.layer_sequence.layers):
            obj_center = reference_points[..., : self.dim].transpose(0, 1)  # [num_queries, batch_size, dim]

            # do not apply transform in position in the first decoder layer
            if idx == 0:
                position_transform = 1
            else:
                position_transform = self.query_scale(query)

            # get sine embedding for the query vector
            query_sine_embed = gen_sine_embed_for_position(obj_center, self.embed_dim)
            # apply position transform
            query_sine_embed = query_sine_embed[..., : self.embed_dim] * position_transform

            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                query_sine_embed=query_sine_embed,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                is_first_layer=(idx == 0),
                **kwargs,
            )

            if self.return_intermediate:
                if self.post_norm_layer is not None:
                    intermediate.append(self.post_norm_layer(query))
                else:
                    intermediate.append(query)

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(query)

        if self.return_intermediate:
            return (
                torch.stack(intermediate),
                reference_points,
            )

        return query.unsqueeze(0), reference_points
