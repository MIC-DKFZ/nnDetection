# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from nndet.nn.transformer.attention.conditional_attention import (
    ConditionalCrossAttention,
    ConditionalSelfAttention,
    _convert_mask,
)

# convert mask


def test_convert_mask_float():
    m = torch.rand(16, 16, dtype=torch.float)
    assert torch.allclose(m, _convert_mask(m, torch.float))


def test_convert_mask_bool():
    m = torch.zeros(16, 16, dtype=torch.bool)
    m[1:3] = True

    expected = torch.zeros(16, 16, dtype=torch.float)
    expected[1:3] = float("-inf")

    assert torch.allclose(expected, _convert_mask(m, torch.float))


##############################
### self attention ###########
##############################

TEST_SETTINGS_SELF = [
    (ConditionalSelfAttention, 64, 4, 0.4, 0.3, False),
    (ConditionalSelfAttention, 512, 8, 0.2, 0.1, False),
]


@pytest.mark.parametrize(
    "attention_module,embed_dim,num_heads,attn_drop_value,proj_drop_value,batch_first",
    TEST_SETTINGS_SELF,
)
def test_conditional_self_attention_settings(
    attention_module,
    embed_dim,
    num_heads,
    attn_drop_value,
    proj_drop_value,
    batch_first,
):
    attention = attention_module(embed_dim, num_heads, attn_drop_value, proj_drop_value, batch_first)
    matrix_shape = torch.Size([embed_dim, embed_dim])
    assert matrix_shape == attention.query_content_proj.weight.shape
    assert matrix_shape == attention.query_pos_proj.weight.shape
    assert matrix_shape == attention.key_content_proj.weight.shape
    assert matrix_shape == attention.key_pos_proj.weight.shape
    assert matrix_shape == attention.value_proj.weight.shape
    assert matrix_shape == attention.out_proj.weight.shape
    assert attention.num_heads == num_heads
    assert attention.attn_drop.p == attn_drop_value
    assert attention.proj_drop.p == proj_drop_value


TEST_SHAPE_SELF = [
    (
        ConditionalSelfAttention(
            embed_dim=512,
            num_heads=8,
        ),
        torch.ones((100, 4, 512)),  # q [N, bs, C]
        torch.ones((100, 4, 512)),  # k
        torch.ones((100, 4, 512)),  # v
        torch.ones((100, 4, 512)),  # qp
        torch.ones((100, 4, 512)),  # kp
        None,  # attn_mask
        None,  # key_padding_mask
        (100, 4, 512),  # expected output shape
    ),
    (
        ConditionalSelfAttention(
            embed_dim=256,
            num_heads=16,
        ),
        torch.ones((80, 8, 256)),  # use different sequence lengths
        torch.ones((80, 8, 256)),  # k
        torch.ones((80, 8, 256)),  # v
        torch.ones((80, 8, 256)),  # q
        torch.ones((80, 8, 256)),  # kp
        None,  # attn_mask
        None,  # key_padding_mask
        (80, 8, 256),  # expected output shape
    ),
    (
        ConditionalSelfAttention(
            embed_dim=256,
            num_heads=16,
        ),
        torch.ones((80, 8, 256)),  # use different sequence lengths
        torch.ones((80, 8, 256)),  # k
        torch.ones((80, 8, 256)),  # v
        torch.ones((80, 8, 256)),  # q
        torch.ones((80, 8, 256)),  # kp
        torch.zeros((80, 80), dtype=torch.bool),  # attn_mask
        None,  # key_padding_mask
        (80, 8, 256),  # expected output shape
    ),
    (
        ConditionalSelfAttention(
            embed_dim=256,
            num_heads=16,
        ),
        torch.ones((80, 8, 256)),  # use different sequence lengths
        torch.ones((80, 8, 256)),  # k
        torch.ones((80, 8, 256)),  # v
        torch.ones((80, 8, 256)),  # q
        torch.ones((80, 8, 256)),  # kp
        None,  # attn_mask
        torch.zeros((8, 80), dtype=torch.bool),  # key_padding_mask
        (80, 8, 256),  # expected output shape
    ),
    (
        ConditionalSelfAttention(
            embed_dim=256,
            num_heads=16,
        ),
        torch.ones((80, 8, 256)),  # use different sequence lengths
        torch.ones((80, 8, 256)),  # k
        torch.ones((80, 8, 256)),  # v
        torch.ones((80, 8, 256)),  # q
        torch.ones((80, 8, 256)),  # kp
        torch.zeros((80, 80), dtype=torch.bool),  # attn_mask
        torch.zeros((8, 80), dtype=torch.bool),  # key_padding_mask
        (80, 8, 256),  # expected output shape
    ),
]


@pytest.mark.parametrize(
    "attention,query,key,value,query_pos,key_pos,attn_mask,key_padding_mask,expected_out_shape",
    TEST_SHAPE_SELF,
)
def test_conditional_self_attention_output_shape(
    attention,
    query,
    key,
    value,
    query_pos,
    key_pos,
    attn_mask,
    key_padding_mask,
    expected_out_shape,
):
    identity = torch.zeros_like(query)
    out = attention(
        query,
        key,
        value,
        identity=identity,
        query_pos=query_pos,
        key_pos=key_pos,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
    )
    assert tuple(out.shape) == expected_out_shape


TEST_SELF_VALUE = [
    (
        ConditionalSelfAttention(
            embed_dim=512,
            num_heads=1,
            bias=False,
        ),
        torch.ones((100, 4, 512)),  # q [N, bs, C]
        torch.ones((100, 4, 512)),  # k
        torch.ones((100, 4, 512)),  # v
        torch.randn((100, 4, 512)),  # identity
        torch.ones((100, 4, 512)),  # qp
        torch.ones((100, 4, 512)),  # kp
    ),
    (
        ConditionalSelfAttention(
            embed_dim=512,
            num_heads=1,
            bias=False,
        ),
        torch.ones((100, 4, 512)),  # q [N, bs, C]
        torch.ones((100, 4, 512)),  # k
        torch.ones((100, 4, 512)),  # v
        torch.randn((100, 4, 512)),  # identity
        torch.zeros((100, 4, 512)),  # qp
        torch.zeros((100, 4, 512)),  # kp
    ),
    (
        ConditionalSelfAttention(
            embed_dim=256,
            num_heads=8,
            bias=False,
        ),
        torch.ones((80, 8, 256)),  # q [N, bs, C]
        torch.ones((80, 8, 256)),  # k
        torch.ones((80, 8, 256)),  # v
        torch.randn((80, 8, 256)),  # identity
        torch.ones((80, 8, 256)),  # qp
        torch.ones((80, 8, 256)),  # kp
    ),
]


@pytest.mark.parametrize("attention,query,key,value,identity,query_pos,key_pos", TEST_SELF_VALUE)
def test_conditional_self_attention_value(attention, query, key, value, identity, query_pos, key_pos):
    for module in attention.modules():
        if isinstance(module, torch.nn.Linear):
            module.weight = torch.nn.Parameter(torch.eye(*module.weight.shape))
            module.bias = torch.nn.Parameter(torch.zeros_like(module.bias))
    out = attention(query, key, value, identity=identity, query_pos=query_pos, key_pos=key_pos)
    # The attention matrix has the same value in every entry so the output depends on the value of value
    out_expected = torch.zeros_like(query)
    torch.fill_(out_expected, value.mean())
    assert torch.allclose(out, out_expected + identity, atol=1e-6)


##############################
### cross attention ##########
##############################

TEST_SETTINGS_CROSS = [
    (ConditionalCrossAttention, 64, 4, 0.4, 0.3, False),
    (ConditionalCrossAttention, 512, 8, 0.2, 0.1, False),
]


@pytest.mark.parametrize(
    "attention_module,embed_dim,num_heads,attn_drop_value,proj_drop_value,batch_first",
    TEST_SETTINGS_CROSS,
)
def test_conditional_cross_attention_settings(
    attention_module,
    embed_dim,
    num_heads,
    attn_drop_value,
    proj_drop_value,
    batch_first,
):
    attention = attention_module(embed_dim, num_heads, attn_drop_value, proj_drop_value, batch_first)
    matrix_shape = torch.Size([embed_dim, embed_dim])
    assert matrix_shape == attention.query_content_proj.weight.shape
    assert matrix_shape == attention.query_pos_proj.weight.shape
    assert matrix_shape == attention.query_pos_sine_proj.weight.shape
    assert matrix_shape == attention.key_content_proj.weight.shape
    assert matrix_shape == attention.key_pos_proj.weight.shape
    assert matrix_shape == attention.value_proj.weight.shape
    assert matrix_shape == attention.out_proj.weight.shape
    assert attention.num_heads == num_heads
    assert attention.attn_drop.p == attn_drop_value
    assert attention.proj_drop.p == proj_drop_value


TEST_SHAPE_CROSS = [
    (
        ConditionalCrossAttention(
            embed_dim=512,
            num_heads=8,
        ),
        torch.ones((100, 4, 512)),  # q [N, bs, C]
        torch.ones((100, 4, 512)),  # k
        torch.ones((100, 4, 512)),  # v
        torch.ones((100, 4, 512)),  # qp
        torch.ones((100, 4, 512)),  # kp
        torch.ones((100, 4, 512)),  # qse
        None,  # attn_mask
        None,  # key_padding_mask
        (100, 4, 512),  # exp shape
    ),
    (
        ConditionalCrossAttention(
            embed_dim=256,
            num_heads=16,
        ),
        torch.ones((32, 8, 256)),  # use different sequence lengths
        torch.ones((80, 8, 256)),  # k
        torch.ones((80, 8, 256)),  # v
        torch.ones((32, 8, 256)),  # qp
        torch.ones((80, 8, 256)),  # kp
        torch.ones((32, 8, 256)),  # qse
        None,  # attn_mask
        None,  # key_padding_mask
        (32, 8, 256),  # exp shape
    ),
    (
        ConditionalCrossAttention(
            embed_dim=256,
            num_heads=16,
        ),
        torch.ones((32, 8, 256)),  # use different sequence lengths
        torch.ones((80, 8, 256)),  # k
        torch.ones((80, 8, 256)),  # v
        torch.ones((32, 8, 256)),  # qp
        torch.ones((80, 8, 256)),  # kp
        torch.ones((32, 8, 256)),  # qse
        torch.zeros((32, 80), dtype=torch.bool),  # attn_mask
        None,  # key_padding_mask
        (32, 8, 256),  # exp shape
    ),
    (
        ConditionalCrossAttention(
            embed_dim=256,
            num_heads=16,
        ),
        torch.ones((32, 8, 256)),  # use different sequence lengths
        torch.ones((80, 8, 256)),  # k
        torch.ones((80, 8, 256)),  # v
        torch.ones((32, 8, 256)),  # qp
        torch.ones((80, 8, 256)),  # kp
        torch.ones((32, 8, 256)),  # qse
        None,  # attn_mask
        torch.zeros((8, 80), dtype=torch.bool),  # key_padding_mask
        (32, 8, 256),  # exp shape
    ),
    (
        ConditionalCrossAttention(
            embed_dim=256,
            num_heads=16,
        ),
        torch.ones((32, 8, 256)),  # q use different sequence lengths
        torch.ones((80, 8, 256)),  # k
        torch.ones((80, 8, 256)),  # v
        torch.ones((32, 8, 256)),  # qp
        torch.ones((80, 8, 256)),  # kp
        torch.ones((32, 8, 256)),  # qse
        torch.zeros((32, 80), dtype=torch.bool),  # attn_mask
        torch.zeros((8, 80), dtype=torch.bool),  # key_padding_mask
        (32, 8, 256),  # exp shape
    ),
]


@pytest.mark.parametrize("is_first_layer", [True, False])
@pytest.mark.parametrize(
    "attention,query,key,value,query_pos,key_pos,query_sine_embed,attn_mask,key_padding_mask,expected_out_shape",
    TEST_SHAPE_CROSS,
)
def test_conditional_cross_attention_shape(
    attention,
    query,
    key,
    value,
    query_pos,
    key_pos,
    query_sine_embed,
    attn_mask,
    key_padding_mask,
    is_first_layer,
    expected_out_shape,
):
    identity = torch.zeros_like(query)
    out = attention(
        query,
        key,
        value,
        identity=identity,
        query_pos=query_pos,
        key_pos=key_pos,
        query_sine_embed=query_sine_embed,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        is_first_layer=is_first_layer,
    )
    assert tuple(out.shape) == expected_out_shape


TEST_CROSS_VALUE = [
    (
        ConditionalCrossAttention(
            embed_dim=512,
            num_heads=1,
            bias=False,
        ),
        torch.ones((100, 4, 512)),  # q [N, bs, C]
        torch.ones((128, 4, 512)),  # k
        torch.ones((128, 4, 512)),  # v
        torch.ones((100, 4, 512)),  # ident
        torch.ones((100, 4, 512)),  # qp
        torch.ones((128, 4, 512)),  # kp
        torch.ones((100, 4, 512)),  # query sine embed
    ),
    (
        ConditionalCrossAttention(
            embed_dim=256,
            num_heads=8,
            bias=False,
        ),
        torch.ones((80, 8, 256)),
        torch.ones((32, 8, 256)),
        torch.ones((32, 8, 256)),
        torch.randn((80, 8, 256)),
        torch.ones((80, 8, 256)),
        torch.ones((32, 8, 256)),
        torch.ones((80, 8, 256)),
    ),
    (
        ConditionalCrossAttention(
            embed_dim=256,
            num_heads=8,
            bias=False,
        ),
        torch.ones((80, 8, 256)),
        torch.ones((32, 8, 256)),
        torch.ones((32, 8, 256)),
        torch.ones((80, 8, 256)),
        torch.zeros((80, 8, 256)),
        torch.zeros((32, 8, 256)),
        torch.zeros((80, 8, 256)),
    ),
    (
        ConditionalCrossAttention(
            embed_dim=256,
            num_heads=8,
            bias=False,
        ),
        torch.zeros((80, 8, 256)),
        torch.zeros((32, 8, 256)),
        torch.zeros((32, 8, 256)),
        torch.ones((80, 8, 256)),
        torch.ones((80, 8, 256)),
        torch.ones((32, 8, 256)),
        torch.ones((80, 8, 256)),
    ),
]


@pytest.mark.parametrize(
    "attention,query,key,value,identity,query_pos,key_pos,query_sine_embed",
    TEST_CROSS_VALUE,
)
def test_conditional_cross_attention_value(
    attention,
    query,
    key,
    value,
    identity,
    query_pos,
    key_pos,
    query_sine_embed,
):
    for module in attention.modules():
        if isinstance(module, torch.nn.Linear):
            module.weight = torch.nn.Parameter(torch.eye(*module.weight.shape))
            module.bias = torch.nn.Parameter(torch.zeros_like(module.bias))
    # This should output query + identity (identity connection) (ONLY GIVEN THE EXACT SETTINGS IN TEST_VALUE)
    out = attention(
        query,
        key,
        value,
        identity=identity,
        query_pos=query_pos,
        key_pos=key_pos,
        query_sine_embed=query_sine_embed,
    )
    if identity is None:
        identity = query
    # The attention matrix has the same value in every entry so the output depends on the value of value
    out_expected = torch.zeros_like(query)
    torch.fill_(out_expected, value.mean())
    assert torch.allclose(out, out_expected + identity, atol=1e-6)
