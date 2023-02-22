# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
from abc import abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class AbstractTransformerEncoder(nn.Module):
    @abstractmethod
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
        Compute a sequence of refined features. Typical inputs are query and query_pos.

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


class AbstractTransformerDecoder(nn.Module):
    @abstractmethod
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
            **kwargs:

        Returns:
            Tensor: Sequence of output embeddings, either of the last layer if
                return_intermediate is false  or of all layers with shape
                ((num_decoder_layers), num_queries, bs, C)
        """
