# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from detrex licensed under
# SPDX-FileCopyrightText: 2022, The IDEA Authors
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Optional

import torch
from torch import nn as nn


class ConditionalSelfAttention(nn.Module):
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
        Conditional Self-Attention Module used in Conditional-DETR

        Args:
            embed_dim: The embedding dimension for attention.
            num_heads: The number of attention heads.
            attn_drop_value: A Dropout layer on attn_output_weights.
            proj_drop_value: A Dropout layer after `MultiheadAttention`.
            batch_first: if `True`, then the input and output tensor will be
                provided as `(bs, n, embed_dim)`
            kwargs: ignored
        """
        super(ConditionalSelfAttention, self).__init__()
        self.query_content_proj = nn.Linear(embed_dim, embed_dim)
        self.query_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.key_content_proj = nn.Linear(embed_dim, embed_dim)
        self.key_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop_value)
        self.proj_drop = nn.Dropout(proj_drop_value)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        head_dim = embed_dim // num_heads
        self.scale = head_dim**-0.5
        self.batch_first = batch_first

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward function for `ConditionalSelfAttention`
        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query: Query embeddings with shape
                `(num_query, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_query, embed_dim)`
            key: Key embeddings with shape
                `(num_key, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_key, embed_dim)`
            value: Value embeddings with the same shape as `key`.
                Same in `torch.nn.MultiheadAttention.forward`.
                If None, the `key` will be used.
            identity: The tensor, with the same shape as `query``,
                which will be used for identity addition.
                If None, `query` will be used.
            query_pos: The position embedding for query, with the
                same shape as `query`.
            key_pos: The position embedding for key.
                If None, and `query_pos` has the same shape as `key`, then
                `query_pos` will be used for `key_pos`.
            attn_mask: ByteTensor mask with shape `(num_query, num_key)`.
                Same as `torch.nn.MultiheadAttention.forward`.
            key_padding_mask: ByteTensor with shape `(bs, num_key)` which
                indicates which elements within `key` to be ignored in
                attention.
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f"position encoding of key is" f"missing in {self.__class__.__name__}.")

        assert (
            query_pos is not None and key_pos is not None
        ), "query_pos and key_pos must be passed into ConditionalAttention Module"

        # transpose (b n c) to (n b c) for attention calculation
        if self.batch_first:
            query = query.transpose(0, 1)  # (n b c)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            query_pos = query_pos.transpose(0, 1)
            key_pos = key_pos.transpose(0, 1)
            identity = identity.transpose(0, 1)

        # query/key/value content and position embedding projection
        query_content = self.query_content_proj(query)
        query_pos = self.query_pos_proj(query_pos)
        key_content = self.key_content_proj(key)
        key_pos = self.key_pos_proj(key_pos)
        value = self.value_proj(value)

        # attention calculation
        N, B, C = query_content.shape
        q = query_content + query_pos
        k = key_content + key_pos
        v = value

        q = q.reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)  # (B, num_heads, N, head_dim)
        k = k.reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        v = v.reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # add attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn.masked_fill_(attn_mask, float("-inf"))
            else:
                attn += attn_mask
        if key_padding_mask is not None:
            attn = attn.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)
        return identity + self.proj_drop(out)


class ConditionalCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        attn_drop_value=0.0,
        proj_drop_value=0.0,
        batch_first=False,
        **kwargs,
    ):
        """
        Conditional Cross-Attention Module used in Conditional-DETR

        Args:
            embed_dim: The embedding dimension for attention.
            num_heads: The number of attention heads.
            attn_drop_value: A Dropout layer on attn_output_weights.
            proj_drop_value: A Dropout layer after `MultiheadAttention`.
            batch_first: if `True`, then the input and output tensor will be
                provided as `(bs, n, embed_dim)`.
            kwargs: ignored
        """

        super(ConditionalCrossAttention, self).__init__()
        self.query_content_proj = nn.Linear(embed_dim, embed_dim)
        self.query_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.query_pos_sine_proj = nn.Linear(embed_dim, embed_dim)
        self.key_content_proj = nn.Linear(embed_dim, embed_dim)
        self.key_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop_value)
        self.proj_drop = nn.Dropout(proj_drop_value)
        self.num_heads = num_heads
        self.batch_first = batch_first

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        query_sine_embed=None,
        is_first_layer=False,
        attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """
        Forward function for `ConditionalCrossAttention`
        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

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
            query_sine_embed: None
            is_first_layer: None
            attn_mask: ByteTensor mask with shape `(num_query, num_key)`. Same
                as `torch.nn.MultiheadAttention.forward`.
            key_padding_mask: ByteTensor with shape `(bs, num_key)` which
                indicates which elements within `key` to be ignored in
                attention.
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f"position encoding of key is" f"missing in {self.__class__.__name__}.")

        assert (
            query_pos is not None and key_pos is not None
        ), "query_pos and key_pos must be passed into ConditionalAttention Module"

        # transpose (b n c) to (n b c) for attention calculation
        if self.batch_first:
            query = query.transpose(0, 1)  # (n b c)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            query_pos = query_pos.transpose(0, 1)
            key_pos = key_pos.transpose(0, 1)
            identity = identity.transpose(0, 1)

        # content projection
        query_content = self.query_content_proj(query)
        key_content = self.key_content_proj(key)
        value = self.value_proj(value)

        # shape info
        N, B, C = query_content.shape
        HW, _, _ = key_content.shape

        # position projection
        key_pos = self.key_pos_proj(key_pos)
        if is_first_layer:
            query_pos = self.query_pos_proj(query_pos)
            q = query_content + query_pos
            k = key_content + key_pos
        else:
            q = query_content
            k = key_content
        v = value

        # preprocess
        q = q.view(N, B, self.num_heads, C // self.num_heads)
        query_sine_embed = self.query_pos_sine_proj(query_sine_embed).view(N, B, self.num_heads, C // self.num_heads)
        q = torch.cat([q, query_sine_embed], dim=3).view(N, B, C * 2)

        k = k.view(HW, B, self.num_heads, C // self.num_heads)  # N, 16, 256
        key_pos = key_pos.view(HW, B, self.num_heads, C // self.num_heads)
        k = torch.cat([k, key_pos], dim=3).view(HW, B, C * 2)

        # attention calculation
        q = q.reshape(N, B, self.num_heads, C * 2 // self.num_heads).permute(1, 2, 0, 3)  # (B, num_heads, N, head_dim)
        k = k.reshape(HW, B, self.num_heads, C * 2 // self.num_heads).permute(1, 2, 0, 3)
        v = v.reshape(HW, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)

        scale = (C * 2 // self.num_heads) ** -0.5
        q = q * scale
        attn = q @ k.transpose(-2, -1)

        # add attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn.masked_fill_(attn_mask, float("-inf"))
            else:
                attn += attn_mask
        if key_padding_mask is not None:
            attn = attn.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.proj_drop(out)
