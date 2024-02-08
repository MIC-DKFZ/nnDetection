# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from detrex licensed under
# SPDX-FileCopyrightText: 2022, The IDEA Authors
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from nndet.nn.layers.mlp import ReluDropIdentityMLP
from nndet.nn.transformer.attention.attention import MultiheadAttention
from nndet.nn.transformer.layers.abstract import (
    BaseTransformerDecoder,
    BaseTransformerEncoder,
)
from nndet.nn.transformer.layers.base_layer import BaseTransformerLayer


class DETRTransformerEncoder(BaseTransformerEncoder):
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
        batch_first: bool = False,
    ):
        """
        Transformer Encoder for DETR. Consists of num_layers transformer encoder
        layers refining the input feature sequence.

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
            dim: dimension of the input, has to be 2 or 3
            batch_first: use batch first computations in the transformer
        """
        super().__init__(embed_dim=embed_dim, dim=dim)

        transformer_layer = BaseTransformerLayer(
            attn=MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_drop_value=attn_dropout,
                proj_drop_value=proj_dropout,
                batch_first=batch_first,
            ),
            ffn=ReluDropIdentityMLP(
                embed_dim=embed_dim,
                feedforward_dim=feedforward_dim,
                ffn_drop=ffn_dropout,
                num_layers=num_ffn_layers,
            ),
            norm=nn.LayerNorm(
                normalized_shape=embed_dim,
            ),
            operation_order=("self_attn", "norm", "ffn", "norm"),
        )
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(copy.deepcopy(transformer_layer))

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
        """
        Compute a sequence of refined features. Typical inputs are query and
        query_pos.

        Args:
            query: sequence of input features (sequence_length, bs, C)
            key: (Optional) key for attention
            value: (Optional) value for attention
            query_pos: (Optional) position embedding for the given query
                (sequence_length, bs, C)
            key_pos: (Optional) position embedding for the given key
            attn_masks: (Optional) mask for the attention layer
            query_key_padding_mask: (Optional) query key padding mask for
                attention
            key_padding_mask: (Optional) key padding mask for attention
            **kwargs:

        Returns:
            Tensor: Sequence of refined features (sequence_length, bs, C)
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


class DETRTransformerDecoder(BaseTransformerDecoder):
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
    ):
        """
        Transformer Decoder for DETR

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
            return_intermediate: return the outputs of all decoder layers
            dim: dimension of the input, has to be 2 or 3
            batch_first: use batch first computations in the transformer
        """
        super().__init__(embed_dim=embed_dim, dim=dim)

        transformer_layer = BaseTransformerLayer(
            attn=MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_drop_value=attn_dropout,
                proj_drop_value=proj_dropout,
                batch_first=batch_first,
            ),
            ffn=ReluDropIdentityMLP(
                embed_dim=embed_dim,
                feedforward_dim=feedforward_dim,
                ffn_drop=ffn_dropout,
                num_layers=num_ffn_layers,
            ),
            norm=nn.LayerNorm(
                normalized_shape=embed_dim,
            ),
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
        attn_masks: Optional[List[torch.Tensor]] = None,
        query_key_padding_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
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
        """

        if not self.return_intermediate:
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
            return query[None], None
        else:
            # return intermediate
            intermediate = []
            for layer in self.layers:
                query = layer(
                    query,
                    key,
                    value,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_masks=attn_masks,
                    query_key_padding_mask=query_key_padding_mask,
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )

                if self.return_intermediate:
                    if self.post_norm_layer is not None:
                        intermediate.append(self.post_norm_layer(query))
                    else:
                        intermediate.append(query)

            return torch.stack(intermediate), None
