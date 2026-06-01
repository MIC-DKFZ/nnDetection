# Modifications licensed under:
# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from detrex licensed under
# SPDX-FileCopyrightText: 2022, The IDEA Authors
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from mmdetection licensed under
# SPDX-FileCopyrightText: 2022, OpenMMLab
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

import torch
import torch.nn as nn
from loguru import logger


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_drop_value: float = 0.0,
        proj_drop_value: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        """
        A wrapper for ``torch.nn.MultiheadAttention``
        Implemented MultiheadAttention with identity connection,
        and position embedding is also passed as input.

        Args:
            embed_dim: The embedding dimension for attention.
            num_heads: The number of attention heads.
            attn_drop_value: A Dropout layer on attn_output_weights.
            proj_drop_value: A Dropout layer after `MultiheadAttention`.
            batch_first: if `True`, then the input and output tensor will be
                provided as `(bs, n, embed_dim)`.
            **kwargs: kwargs for nn.MultiheadAttention, could include
                'add_bias_kv', 'kdim', 'vdim'
        """
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop_value,
            batch_first=batch_first,
            **kwargs,
        )

        self.proj_drop = nn.Dropout(proj_drop_value)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        identity: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward function for `MultiheadAttention`. **kwargs allow passing a more
        general data flow when combining with other operations in
        `transformerlayer`.

        Args:
            query: Query embeddings with shape `(num_query, bs, embed_dim)` if
                self.batch_first is False, else `(bs, num_query, embed_dim)`
            key: Key embeddings with shape `(num_key, bs, embed_dim)` if
                self.batch_first is False, else `(bs, num_key, embed_dim)`
            value: Value embeddings with the same shape as `key`. Same in
                `torch.nn.MultiheadAttention.forward`. If None, the `key` will
                be used.
            identity: The tensor, with the same shape as x, will be used for
                identity addition. If None, `query` will be used.
            query_pos: The position embedding for query, with the same shape as
                `query`.
            key_pos: The position embedding for key. If None, and `query_pos`
                has the same shape as `key`, then `query_pos` will be used for
                `key_pos`.
            attn_mask: ByteTensor mask with shape `(num_query, num_key)`. Same
                as `torch.nn.MultiheadAttention.forward`.
            key_padding_mask: ByteTensor with shape `(bs, num_key)` which
                indicates which elements within `key` to be ignored in
                attention.

        Returns:
            the output sequence with shape `(num_query, bs, embed_dim)`
        """
        assert identity is not None
        if query_pos is None and key_pos is None:
            logger.warning(f"position encoding of query and key is" f"missing in {self.__class__.__name__}.")

        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]

        return identity + self.proj_drop(out)
