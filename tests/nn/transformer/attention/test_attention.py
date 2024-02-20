# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from nndet.nn.transformer.attention.attention import MultiheadAttention

TEST_SETTINGS = [
    (64, 4, 0.4, 0.3, False),
    (512, 8, 0.2, 0.1, False),
]

TEST_SHAPE = [
    (
        MultiheadAttention(
            embed_dim=512,
            num_heads=8,
        ),
        torch.ones((100, 4, 512)),  # [N, bs, C]
        torch.ones((100, 4, 512)),
        torch.ones((100, 4, 512)),
        (100, 4, 512),
    ),
    (
        MultiheadAttention(
            embed_dim=256,
            num_heads=16,
        ),
        torch.ones((32, 8, 256)),  # use different sequence lengths
        torch.ones((80, 8, 256)),
        torch.ones((80, 8, 256)),
        (32, 8, 256),
    ),
]


@pytest.mark.parametrize("embed_dim,num_heads,attn_drop_value,proj_drop_value,batch_first", TEST_SETTINGS)
def test_multi_head_attention_settings(embed_dim, num_heads, attn_drop_value, proj_drop_value, batch_first):
    attention = MultiheadAttention(embed_dim, num_heads, attn_drop_value, proj_drop_value, batch_first)
    assert attention.embed_dim == attention.attn.embed_dim == embed_dim
    assert attention.num_heads == attention.attn.num_heads == num_heads
    assert attention.attn.dropout == attn_drop_value
    assert attention.proj_drop.p == proj_drop_value


@pytest.mark.parametrize("attention,query,key,value,expected_out_shape", TEST_SHAPE)
def test_multi_head_attention_output_shape(attention, query, key, value, expected_out_shape):
    out = attention(query=query, key=key, value=value, identity=torch.zeros_like(query))
    assert tuple(out.shape) == expected_out_shape


TEST_VALUE = [
    # (
    #     MultiheadAttention(
    #         embed_dim=512,
    #         num_heads=1,
    #         bias=False,
    #     ),
    #     torch.ones((100, 4, 512)),  # [N, bs, C]
    #     torch.ones((100, 4, 512)),
    #     torch.ones((100, 4, 512)),
    #     None,
    # ), # identity = None not supported anymore
    (
        MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            bias=False,
        ),
        torch.ones((32, 1, 256)),  # use different sequence lengths
        torch.ones((80, 1, 256)),
        torch.ones((80, 1, 256)),
        torch.randn((32, 1, 256)),
    ),
]


@pytest.mark.parametrize("attention,query,key,value,identity", TEST_VALUE)
def test_multi_head_attention_value(attention, query, key, value, identity):
    embed_dim = query.shape[-1]
    attention.attn.in_proj_weight = torch.nn.Parameter(
        torch.cat(
            [
                torch.eye(embed_dim, embed_dim),
                torch.eye(embed_dim, embed_dim),
                torch.eye(embed_dim, embed_dim),
            ]
        )
    )
    attention.attn.out_proj.weight = torch.nn.Parameter(torch.eye(embed_dim, embed_dim))
    # This should output query + identity (identity connection) (ONLY GIVEN THE EXACT SETTINGS IN TEST_VALUE)
    out = attention(
        query=query,
        key=key,
        value=value,
        identity=identity,
    )
    if identity is None:
        identity = query
    # The attention matrix has the same value in every entry so the output depends on the value of value
    out_expected = torch.zeros_like(query)
    torch.fill_(out_expected, value.mean())
    assert torch.allclose(out, out_expected + identity, atol=1e-6)
