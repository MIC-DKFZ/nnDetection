# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from detrex licensed under
# SPDX-FileCopyrightText: 2022, The IDEA Authors
# SPDX-License-Identifier: Apache-2.0
import copy
from typing import Optional

import torch
from torch import nn as nn

import nndet.core.ops_torch as ops_torch
from nndet.nn.heads.regressor.ffn import FFNRegressor
from nndet.nn.layers.mlp import ReluDropIdentityMLP
from nndet.nn.transformer.attention.attention import MultiheadAttention
from nndet.nn.transformer.attention.multi_scale_deform_attn_3d import (
    MultiScaleDeformableAttention,
)
from nndet.nn.transformer.layers.abstract import (
    BaseTransformerDecoder,
    BaseTransformerEncoder,
)
from nndet.nn.transformer.layers.base_layer import BaseTransformerLayer


class DeformableDETRTransformerEncoder(BaseTransformerEncoder):
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
        post_norm: bool = False,
        dim: int = 3,
        batch_first: bool = True,
        num_feature_levels: int = 4,
        num_points: int = 4,
    ):
        super().__init__(
            embed_dim=embed_dim,
            dim=dim,
        )

        transformer_layer = BaseTransformerLayer(
            attn=[
                MultiScaleDeformableAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=attn_dropout,
                    batch_first=batch_first,
                    num_levels=num_feature_levels,
                    num_points=num_points,
                )
            ],
            ffn=ReluDropIdentityMLP(
                embed_dim=embed_dim,
                feedforward_dim=feedforward_dim,
                ffn_drop=ffn_dropout,
                num_layers=num_ffn_layers,
            ),
            norm=nn.LayerNorm(embed_dim),
            operation_order=("self_attn", "norm", "ffn", "norm"),
        )
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(copy.deepcopy(transformer_layer))

        self.embed_dim = embed_dim
        self.pre_norm = self.layers[0].pre_norm

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
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


class DeformableDETRTransformerDecoder(BaseTransformerDecoder):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        feedforward_dim: int = 1024,
        num_ffn_layers: int = 2,
        return_intermediate: bool = True,
        dim: int = 3,
        batch_first: bool = True,
        num_feature_levels: int = 4,
        num_points: int = 4,
        regressor: Optional[FFNRegressor] = None,
    ):
        super().__init__(
            embed_dim=embed_dim,
            dim=dim,
        )

        transformer_layer = BaseTransformerLayer(
            attn=[
                MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_drop_value=attn_dropout,
                    proj_drop_value=proj_dropout,
                    batch_first=batch_first,
                ),
                MultiScaleDeformableAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=attn_dropout,
                    batch_first=True,
                    num_levels=num_feature_levels,
                    num_points=num_points,
                ),
            ],
            ffn=ReluDropIdentityMLP(
                embed_dim=embed_dim,
                feedforward_dim=feedforward_dim,
                ffn_drop=ffn_dropout,
                num_layers=num_ffn_layers,
            ),
            norm=nn.LayerNorm(embed_dim),
            operation_order=(
                "self_attn",
                "norm",
                "cross_attn",
                "norm",
                "ffn",
                "norm",
            ),
        )
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(copy.deepcopy(transformer_layer))

        self.return_intermediate = return_intermediate
        self.regressor = regressor

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        reference_points=None,  # num_queries, 4. normalized.
        valid_ratios=None,
        **kwargs,
    ):
        output = query

        intermediate = []
        intermediate_reference_points = []
        for layer_idx, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 6:
                reference_points_input = (
                    reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 3
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points_input,
                **kwargs,
            )

            if self.regressor is not None:
                tmp = self.regressor(output, layer_idx)
                # FIXME the order xyz,whd might be wrong here
                if reference_points.shape[-1] == 6:
                    new_reference_points = tmp + ops_torch.inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 3
                    new_reference_points = tmp
                    new_reference_points[..., :3] = tmp[..., :3] + ops_torch.inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points
