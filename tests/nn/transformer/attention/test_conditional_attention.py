import pytest
import torch

from nndet.nn.transformer.attention.conditional_attention import (
    ConditionalCrossAttention,
    ConditionalSelfAttention,
    _convert_mask,
)

TEST_SETTINGS_SELF = [
    (ConditionalSelfAttention, 64, 4, 0.4, 0.3, False),
    (ConditionalSelfAttention, 512, 8, 0.2, 0.1, False),
]

TEST_SHAPE_SELF = [
    (
        ConditionalSelfAttention(
            embed_dim=512,
            num_heads=8,
        ),
        torch.ones((100, 4, 512)),  # [N, bs, C]
        torch.ones((100, 4, 512)),
        torch.ones((100, 4, 512)),
        torch.ones((100, 4, 512)),
        None,
        (100, 4, 512),
    ),
    (
        ConditionalSelfAttention(
            embed_dim=256,
            num_heads=16,
        ),
        torch.ones((80, 8, 256)),  # use different sequence lengths
        torch.ones((80, 8, 256)),
        torch.ones((80, 8, 256)),
        torch.ones((80, 8, 256)),
        torch.ones((80, 8, 256)),
        (80, 8, 256),
    ),
]


@pytest.mark.parametrize(
    "attention_module,embed_dim,num_heads,attn_drop_value,proj_drop_value,batch_first", TEST_SETTINGS_SELF
)
def test_conditional_self_attention_settings(
    attention_module, embed_dim, num_heads, attn_drop_value, proj_drop_value, batch_first
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


@pytest.mark.parametrize("attention,query,key,value,query_pos,key_pos,expected_out_shape", TEST_SHAPE_SELF)
def test_conditional_self_attention_output_shape(attention, query, key, value, query_pos, key_pos, expected_out_shape):
    out = attention(query, key, value, query_pos=query_pos, key_pos=key_pos)
    assert tuple(out.shape) == expected_out_shape


TEST_SETTINGS_CROSS = [
    (ConditionalCrossAttention, 64, 4, 0.4, 0.3, False),
    (ConditionalCrossAttention, 512, 8, 0.2, 0.1, False),
]

TEST_SHAPE_CROSS = [
    (
        ConditionalCrossAttention(
            embed_dim=512,
            num_heads=8,
        ),
        torch.ones((100, 4, 512)),  # [N, bs, C]
        torch.ones((100, 4, 512)),
        torch.ones((100, 4, 512)),
        torch.ones((100, 4, 512)),
        torch.ones((100, 4, 512)),
        torch.ones((100, 4, 512)),
        (100, 4, 512),
    ),
    (
        ConditionalCrossAttention(
            embed_dim=256,
            num_heads=16,
        ),
        torch.ones((32, 8, 256)),  # use different sequence lengths
        torch.ones((80, 8, 256)),
        torch.ones((80, 8, 256)),
        torch.ones((32, 8, 256)),
        torch.ones((80, 8, 256)),
        torch.ones((32, 8, 256)),
        (32, 8, 256),
    ),
]


@pytest.mark.parametrize(
    "attention_module,embed_dim,num_heads,attn_drop_value,proj_drop_value,batch_first", TEST_SETTINGS_CROSS
)
def test_conditional_cross_attention_settings(
    attention_module, embed_dim, num_heads, attn_drop_value, proj_drop_value, batch_first
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


@pytest.mark.parametrize(
    "attention,query,key,value,query_pos,key_pos,query_sine_embed,expected_out_shape", TEST_SHAPE_CROSS
)
def test_conditional_cross_attention_shape(
    attention, query, key, value, query_pos, key_pos, query_sine_embed, expected_out_shape
):
    out = attention(query, key, value, query_pos=query_pos, key_pos=key_pos, query_sine_embed=query_sine_embed)
    assert tuple(out.shape) == expected_out_shape
