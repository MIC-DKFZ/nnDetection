# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from nndet.nn.transformer.attention.attention import (
    ConditionalCrossAttention,
    ConditionalSelfAttention,
    MultiheadAttention,
)
from nndet.nn.transformer.transformer_base_layer import (
    BaseTransformerLayer,
    TransformerLayerSequence,
)
from nndet.utils.mlp import FFN, MLP


def gen_sineembed3d_for_position(
    pos_tensor: torch.Tensor, num_pos_feats: int, temperature: int = 10000
) -> torch.Tensor:
    """
    3D Positional Encoding to encode given positions (different to the normal position encoding which computes position
    based on pixels)
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

    if num_pos_feats % dim == 0:
        return torch.cat((pos_x, pos_y), dim=2)
    return torch.cat((pos_x, pos_y[:, :, :-1]), dim=2)


class ConditionalDetrTransformerEncoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        attn_dropout: float = 0.1,
        feedforward_dim: int = 2048,
        ffn_dropout: float = 0.1,
        activation: nn.Module = nn.ReLU(),
        post_norm: bool = False,
        dim: int = 3,
        batch_first: bool = False,
    ):
        super(ConditionalDetrTransformerEncoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_drop=attn_dropout,
                    batch_first=batch_first,
                ),
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    ffn_drop=ffn_dropout,
                    activation=activation,
                ),
                norm=nn.LayerNorm(normalized_shape=embed_dim),
                operation_order=("self_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_masks: Optional[List[torch.Tensor]] = None,
        query_key_padding_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class ConditionalDetrTransformerDecoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        attn_dropout: float = 0.1,
        feedforward_dim: int = 2048,
        ffn_dropout: float = 0.1,
        activation: nn.Module = nn.ReLU(),
        post_norm: bool = True,
        return_intermediate: bool = True,
        dim: int = 3,
        batch_first: bool = False,
    ):
        super(ConditionalDetrTransformerDecoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=[
                    ConditionalSelfAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=batch_first,
                    ),
                    ConditionalCrossAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=batch_first,
                    ),
                ],
                ffn=FFN(
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
        self.embed_dim = self.layers[0].embed_dim
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)
        self.ref_point_head = MLP(self.embed_dim, self.embed_dim, dim, 2)
        self.dim = dim
        self.bbox_embed = None

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

        for idx in range(num_layers - 1):
            self.layers[idx + 1].attentions[1].query_pos_proj = None

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_masks: Optional[List[torch.Tensor]] = None,
        query_key_padding_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        intermediate = []
        reference_points_before_sigmoid = self.ref_point_head(query_pos)  # [num_queries, batch_size, dim]
        reference_points = reference_points_before_sigmoid.sigmoid().transpose(0, 1)

        for idx, layer in enumerate(self.layers):
            obj_center = reference_points[..., : self.dim].transpose(0, 1)  # [num_queries, batch_size, dim]

            # do not apply transform in position in the first decoder layer
            if idx == 0:
                position_transform = 1
            else:
                position_transform = self.query_scale(query)

            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed3d_for_position(obj_center, self.embed_dim)
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

        return query.unsqueeze(0)


class ConditionalDetrTransformer(nn.Module):
    def __init__(self, encoder=None, decoder=None):
        super(ConditionalDetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dim = self.encoder.embed_dim

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        features: List[torch.Tensor],
        query_embed: List[torch.Tensor],
        pos_embed: List[torch.Tensor],
        mask: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[Union[torch.Tensor, None], ...]:

        # This is single resolution so unpack the multiscale lists
        assert len(features) == len(query_embed) == len(pos_embed) == 1
        features, query_embed, pos_embed = features[0], query_embed[0], pos_embed[0]

        dim = self.decoder.dim
        assert dim == features.dim() - 2  # subtract batch size and sequence length
        if dim == 2:
            bs, c, h, w = features.shape
        elif dim == 3:
            bs, c, h, w, z = features.shape
        else:
            raise "Number of feature dimensions not supported"

        features = features.view(bs, c, -1).permute(2, 0, 1)  # [bs, c, h, w, (z)] -> [h*w(*z), bs, c]
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]

        if mask is not None:
            assert len(mask) == 0
            mask = mask[0].view(bs, -1)  # [bs, h, w] -> [bs, h*w]

        memory = self.encoder(
            query=features,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask,
        )
        target = torch.zeros_like(query_embed)

        hidden_state, references = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
        )
        hidden_state = hidden_state.transpose(1, 2)
        if dim == 4:
            memory = memory.permute(1, 2, 0).reshape(bs, c, h, w)
        else:
            memory = memory.permute(1, 2, 0).reshape(bs, c, h, w, z)

        return hidden_state, memory, references
