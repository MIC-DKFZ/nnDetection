# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from detrex licensed under
# SPDX-FileCopyrightText: 2022, The IDEA Authors
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional

import torch
from torch import nn as nn


def _convert_mask(
    mask: Optional[torch.Tensor],
    target_type: torch.dtype,
) -> Optional[torch.Tensor]:
    """
    Check if the mask is None or contains bools or floats. If not, raise error.
    If bool, fill the mask with -inf. If float, keep it.

    Args:
        mask: mask for the scaled dot-product attention

    Returns:
        Mask converted to floats or None
    """
    if mask is not None:
        is_float = torch.is_floating_point(mask)
        if mask.dtype != torch.bool and not is_float:
            raise AssertionError("Only bool and floating types of attention masks are supported!")
        if not is_float:
            mask = torch.zeros_like(mask, dtype=target_type).masked_fill_(mask, float("-inf"))
    return mask


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
        self.scale = math.sqrt(embed_dim // num_heads)
        self.batch_first = batch_first
        assert embed_dim % num_heads == 0, f"embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        identity: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
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
        assert identity is not None
        assert (
            query_pos is not None and key_pos is not None
        ), "query_pos and key_pos must be passed into ConditionalAttention Module"

        if self.batch_first:
            # transpose (B, N, C) to (N, B, C) for attention calculation
            query = query.transpose(0, 1)
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

        # Check for masks and convert
        attn_mask = _convert_mask(attn_mask, query_content.dtype)
        key_padding_mask = _convert_mask(key_padding_mask, query_content.dtype)

        # attention calculation
        N, B, C = query_content.shape
        q = query_content + query_pos
        k = key_content + key_pos
        v = value

        # Split into num_heads heads and permute to batch first
        # (N, B, C) -> (N, B, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        q = q.reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        k = k.reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        v = v.reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)

        # merge key padding (B, N) and attention masks
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        q = q / self.scale  # (B, num_heads, N, head_dim)
        attn = q @ k.transpose(-2, -1)

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, C)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)  # (B, N, C)

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
        self.scale = math.sqrt((embed_dim * 2) // num_heads)
        self.batch_first = batch_first
        assert embed_dim % num_heads == 0, f"embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        identity: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
        query_sine_embed: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_first_layer: bool = False,
        **kwargs,
    ) -> torch.Tensor:
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
            query_sine_embed: positional encoding of the center points used for
                the positional part in the cross attention
                with shape `(num_query, bs, embed_dim)`
            attn_mask: ByteTensor mask with shape `(num_query, num_key)`. Same
                as `torch.nn.MultiheadAttention.forward`.
            key_padding_mask: ByteTensor with shape `(bs, num_key)` which
                indicates which elements within `key` to be ignored in
                attention.
            is_first_layer: bool whether its the first decoder layer
        """
        assert identity is not None
        assert (
            query_pos is not None and key_pos is not None
        ), "query_pos and key_pos must be passed into ConditionalAttention Module"

        if self.batch_first:
            # transpose (B, N, C) to (N, B, C) for attention calculation
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            query_pos = query_pos.transpose(0, 1)
            key_pos = key_pos.transpose(0, 1)
            identity = identity.transpose(0, 1)

        # content projection
        query_content = self.query_content_proj(query)  # (N, B, C)
        key_content = self.key_content_proj(key)  # (X, B, C)
        value = self.value_proj(value)  # (X, B, C)

        # shape info
        N, B, C = query_content.shape
        XYZ, _, _ = key_content.shape

        # position projection
        key_pos = self.key_pos_proj(key_pos)
        if is_first_layer:
            query_pos = self.query_pos_proj(query_pos)  # (N, B, C)
            q = query_content + query_pos  # (N, B, C)
            k = key_content + key_pos  # (X, B, C)
        else:
            q = query_content  # (N, B, C)
            k = key_content  # (X, B, C)
        v = value  # (X, B, C)

        # Check for masks and convert
        attn_mask = _convert_mask(attn_mask, q.dtype)
        key_padding_mask = _convert_mask(key_padding_mask, q.dtype)

        # preprocess
        q = q.view(N, B, self.num_heads, C // self.num_heads)  # (N, B, num_heads, head_dim)
        query_sine_embed = self.query_pos_sine_proj(query_sine_embed).view(N, B, self.num_heads, C // self.num_heads)
        q = torch.cat([q, query_sine_embed], dim=3).view(N, B, C * 2)  # (N, B, C * 2)

        k = k.view(XYZ, B, self.num_heads, C // self.num_heads)  # (X, B, num_heads, head_dim)
        key_pos = key_pos.view(XYZ, B, self.num_heads, C // self.num_heads)
        k = torch.cat([k, key_pos], dim=3).view(XYZ, B, C * 2)  # (X, B, C * 2)

        # attention calculation
        # (N, B, C) -> (N, B, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        q = q.reshape(N, B, self.num_heads, C * 2 // self.num_heads).permute(1, 2, 0, 3)
        k = k.reshape(XYZ, B, self.num_heads, C * 2 // self.num_heads).permute(1, 2, 0, 3)
        v = v.reshape(XYZ, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)

        # merge key padding (B, N) and attention masks
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        q = q / self.scale  # (B, num_heads, N, head_dim)
        attn = q @ k.transpose(-2, -1)  # B, num_heads, N, X

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, C)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.proj_drop(out)
