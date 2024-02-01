# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from detrex licensed under
# SPDX-FileCopyrightText: 2022, The IDEA Authors
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Optional, Tuple

import torch
from torch import nn as nn

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
        num_feature_levels: int = 4,  # TODO: add to config
        num_points: int = 4,  # TODO: add to config
    ):
        """
        Transformer Encoder for Deformable DETR Model

        Args:
            embed_dim: embed dimension (hidden dimension) of the transformer
                decoder
            num_heads: number of attention heads
            num_layers: number of decoder layers
            attn_dropout: dropout in the attention modules
            proj_dropout: dropout of the final linear projection after attention
                Not used here! Use `attn_dropout` instead.
            feedforward_dim: hidden dimension of the feed forward network in the
                transformer layer
            ffn_dropout: dropout of the feed forward network
            num_ffn_layers: number of layers in the transformer ffn
            post_norm: apply an additional layer norm to all outputs
            dim: dimension of the input, has to be 2 or 3
            batch_first: use batch first computations in the transformer
            num_feature_levels: number of feature levels used for multi-scale
                attention
            num_points: number of sampling points for each query
        """
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
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_masks: Optional[torch.Tensor] = None,
        query_key_padding_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Perform forward pass through the transformer encoders

        Args:
            query: Query embeddings with shape `(num_query, bs, embed_dim)`
            key: Key embeddings with shape `(num_key, bs, embed_dim)`
            value: Value embeddings with shape `(num_key, bs, embed_dim)`
            query_pos: The position embedding for `query`. Default: None.
            key_pos: (Optional) position embedding for the given key
            attn_masks: (Optional) mask for the attention layer
            query_key_padding_mask: (Optional) query key padding mask for
                attention
            key_padding_mask: (Optional) key padding mask for attention

        Returns:
            torch.Tensor: processed features (seq_length, bs, C)
        """
        for layer in self.layers:
            query = layer(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
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
        """
        Transformer Decoder for Deformable DETR Model

        Args:
            embed_dim: embed dimension (hidden dimension) of the transformer
                decoder
            num_heads: number of attention heads
            num_layers: number of decoder layers
            attn_dropout: dropout in the attention modules
            proj_dropout: dropout of the final linear projection after attention
                Not used here! Use `attn_dropout` instead.
            ffn_dropout: dropout of the feed forward network
            feedforward_dim: hidden dimension of the feed forward network in the
                transformer layer
            num_ffn_layers: number of layers in the transformer ffn
            return_intermediate: return the outputs of all decoder layers
            dim: dimension of the input, has to be 2 or 3
            batch_first: use batch first computations in the transformer
            num_feature_levels: number of feature levels used for multi-scale
                attention
            num_points: number of sampling points for each query
            regressor: regressor to update boxes/points for iterative box
                refinement
        """
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
                    batch_first=batch_first,
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
        self.num_feature_levels = num_feature_levels

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        reference_points: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_masks: Optional[torch.Tensor] = None,
        query_key_padding_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform forward pass through the transformer encoders

        Args:
            query: Query embeddings with shape `(num_query, bs, embed_dim)`
            key: Key embeddings with shape `(num_key, bs, embed_dim)`
            value: Value embeddings with shape `(num_key, bs, embed_dim)`
            reference_points: reference points for deformable attention
                of shape: `(bs, num_queries, 2 * dims)` [`two_stage` enabled]
                or `(bs, num_queries, dims)`  [`two_stage` disabled].
                Reference points need to be normalized and in format center
                format of shape `cx, cy, cz` [`two_stage` disabled] or
                `cx, cy, cz, dx, dy, dz` [`two_stage` enabled]
            query_pos: The position embedding for `query`. Default: None.
            key_pos: (Optional) position embedding for the given key
            attn_masks: (Optional) mask for the attention layer
            query_key_padding_mask: (Optional) query key padding mask for
                attention
            key_padding_mask: (Optional) key padding mask for attention

        Returns:
            torch.Tensor: processed features of shape
                `(num_layers, bs, num_queries, embed_dim)` where num_layers
                is the number of decoder layers, bs is the batch size and
                num_queries is the number of queries (aka predictions).
            torch.Tensor: intermediate reference points after regression
                prediction (num_layers, bs, num_queries, 2 * dims).
                Reference points are in center format
                of shape `cx, cy, cz` [`two_stage` disabled  & `regressor=None`]
                or `cx, cy, cz, dx, dy, dz` [`two_stage` enabled] and
                normalized.
        """
        output = query

        intermediate = []
        intermediate_reference_points = []
        for layer_idx, layer in enumerate(self.layers):
            assert reference_points.shape[-1] in [self.dim, self.dim * 2]
            reference_points_input = reference_points[:, :, None].expand(-1, -1, self.num_feature_levels, -1)

            output = layer(
                query=output,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points_input,
                **kwargs,
            )

            if self.regressor is not None:
                tmp = self.regressor(output, layer_idx)  # bs, num_queries, 2 * dims
                if reference_points.shape[-1] == self.dim * 2:
                    new_reference_points = tmp + self.regressor.apply_inverse_non_lin(reference_points)
                    new_reference_points = self.regressor.apply_non_lin(new_reference_points)
                else:
                    assert reference_points.shape[-1] == self.dim
                    new_reference_points = tmp
                    new_reference_points[..., : self.dim] = tmp[..., : self.dim] + self.regressor.apply_inverse_non_lin(
                        reference_points
                    )
                    new_reference_points = self.regressor.apply_non_lin(new_reference_points)
                reference_points = new_reference_points.detach()  # stop gradient

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points
