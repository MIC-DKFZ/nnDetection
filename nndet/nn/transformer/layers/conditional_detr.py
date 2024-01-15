# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from detrex licensed under
# SPDX-FileCopyrightText: 2022, The IDEA Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from nndet.nn.heads.regressor.ffn import FFNRegressor
from nndet.nn.layers.mlp import ReluDropIdentityMLP, ReluMLP
from nndet.nn.transformer.attention.conditional_attention import (
    ConditionalCrossAttention,
    ConditionalSelfAttention,
)
from nndet.nn.transformer.layers.abstract import BaseTransformerDecoder
from nndet.nn.transformer.layers.base_layer import BaseTransformerLayer


def gen_sine_embed_for_position(
    pos_tensor: torch.Tensor,
    num_pos_feats: int,
    temperature: int = 10000,
) -> torch.Tensor:
    """
    2D or 3D Positional Encoding to encode given positions (different to the
    normal position encoding which computes position based on pixels)

    Args:
        pos_tensor: tensor of shape `(bs, num_pos, dim)`
        num_pos_feats: number of out features (output dimension)
        temperature: temperature of the position encoding

    Returns:
        Tensor: tensor containing position embedding
            `(bs, num_pos, num_pos_feats)`
    """
    dim = pos_tensor.shape[2]
    assert dim in [2, 3]

    scale = 2 * math.pi
    feats = 2 * math.ceil(num_pos_feats / (2 * dim))

    dim_t = torch.arange(feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / feats)

    x_embed = pos_tensor[:, :, 0] * scale  # [batch_size, num_pos]
    y_embed = pos_tensor[:, :, 1] * scale  # [batch_size, num_pos]

    pos_x = x_embed[:, :, None] / dim_t  # [batch_size, num_pos, feats]
    pos_y = y_embed[:, :, None] / dim_t  # [batch_size, num_pos, feats]
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

    # Handle 3D Case
    if dim == 3:
        z_embed = pos_tensor[:, :, 2] * scale
        pos_z = z_embed[:, :, None] / dim_t
        pos_z = torch.stack((pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=3).flatten(2)

        # If num_pos_feats is not divisible by 3 we have to remove some values
        dimension_delta = dim * feats - num_pos_feats
        cut = feats

        if dimension_delta >= 3:
            cut = feats - 1

        if dimension_delta % dim == 0:
            pos_embed = torch.cat((pos_x[:, :, :cut], pos_y[:, :, :cut], pos_z[:, :, :cut]), dim=2)
        elif dimension_delta % dim == 1:
            pos_embed = torch.cat((pos_x[:, :, :cut], pos_y[:, :, :cut], pos_z[:, :, : cut - 1]), dim=2)
        else:
            pos_embed = torch.cat(
                (pos_x[:, :, :cut], pos_y[:, :, : cut - 1], pos_z[:, :, : cut - 1]),
                dim=2,
            )
    else:  # 2D Case
        dimension_delta = dim * feats - num_pos_feats
        cut = feats

        if dimension_delta >= 2:
            cut = feats - 1

        if num_pos_feats % dim == 0:
            pos_embed = torch.cat((pos_x[:, :, :cut], pos_y[:, :, :cut]), dim=2)
        else:
            pos_embed = torch.cat((pos_x[:, :, :cut], pos_y[:, :, : cut - 1]), dim=2)
    return pos_embed


class ConditionalDETRTransformerDecoder(BaseTransformerDecoder):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        feedforward_dim: int = 2048,
        ffn_dropout: float = 0.1,
        num_ffn_layers: int = 2,
        post_norm: bool = True,
        return_intermediate: bool = True,
        dim: int = 3,
        batch_first: bool = False,
        ffn_regressor_cls: FFNRegressor = FFNRegressor,
        temperature: int = 10000,
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
            num_ffn_layers: number of layers in the transformer ffn
            post_norm: apply an additional layer norm to all outputs
            return_intermediate: return the outputs of all
            dim: dimension of the input, has to be 2 or 3
            batch_first: use batch first computations in the transformer
            reg_point_norm_fn: module to normalise the reference point
            temperature: temperature factor for computig positional encoding
        """
        super().__init__(embed_dim=embed_dim, dim=dim)

        transformer_layer = BaseTransformerLayer(
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
            ffn=ReluDropIdentityMLP(
                embed_dim=embed_dim,
                feedforward_dim=feedforward_dim,
                ffn_drop=ffn_dropout,
                num_layers=num_ffn_layers,
            ),
            norm=nn.LayerNorm(
                normalized_shape=embed_dim,
            ),
            operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
        )
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(copy.deepcopy(transformer_layer))

        self.temperature = temperature

        self.return_intermediate = return_intermediate
        self.query_scale = ReluMLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)
        self.ref_point_head = ReluMLP(self.embed_dim, self.embed_dim, dim, 2)
        self.ffn_regressor_cls = ffn_regressor_cls

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

        for idx in range(num_layers - 1):
            self.layers[idx + 1].attentions[1].query_pos_proj = None

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
            **kwargs: kwargs for the transformer layers

        Returns:
            Tensor: Sequence of output embeddings, either of the last layer if
                return_intermediate is false  or of all layers with shape
                ((num_decoder_layers), num_queries, bs, C)

            Tensor: normalized (!) reference points of shape
                [num_queries, batch_size, dim]
        """

        intermediate = []
        reference_points_before_sigmoid = self.ref_point_head(query_pos)  # [num_queries, batch_size, dim]
        assert reference_points_before_sigmoid.shape[-1] == self.dim
        reference_points = self.ffn_regressor_cls.apply_non_lin(reference_points_before_sigmoid)

        for idx, layer in enumerate(self.layers):
            # do not apply transform in position in the first decoder layer
            if idx == 0:
                position_transform = 1
            else:
                position_transform = self.query_scale(query)  # [num_queries, batch_size, embed_dim]

            # get sine embedding for the query vector
            query_sine_embed = gen_sine_embed_for_position(
                reference_points,  # reference_points = obj_center
                num_pos_feats=self.embed_dim,
                temperature=self.temperature,
            )  # [num_queries, batch_size, embed_dim]

            # apply position transform
            query_sine_embed = query_sine_embed * position_transform  # [num_queries, batch_size, embed_dim]

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
                reference_points.transpose(0, 1),  # [batch_size, num_queries, dim]
            )
        else:
            return query.unsqueeze(0), reference_points.transpose(0, 1)  # [batch_size, num_queries, dim]
