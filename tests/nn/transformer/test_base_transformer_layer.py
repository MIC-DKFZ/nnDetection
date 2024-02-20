# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch.nn

from nndet.nn.layers.mlp import ReluDropIdentityMLP
from nndet.nn.transformer.attention.attention import MultiheadAttention
from nndet.nn.transformer.layers.base_layer import BaseTransformerLayer

# test if all modules are accessed
# test if output shape is correct

TEST_CASES_BASE_LAYER = [
    (
        torch.ones((100, 4, 256)),
        BaseTransformerLayer(
            attn=MultiheadAttention(256, 8),
            ffn=ReluDropIdentityMLP(256, 1024, 2),
            norm=torch.nn.LayerNorm(normalized_shape=256),
            operation_order=("self_attn", "norm", "ffn", "norm"),
        ),
    ),
    (
        torch.ones((100, 4, 120)),
        BaseTransformerLayer(
            attn=[MultiheadAttention(120, 5), MultiheadAttention(120, 5)],
            ffn=ReluDropIdentityMLP(120, 1024, 2),
            norm=torch.nn.LayerNorm(normalized_shape=120),
            operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
        ),
    ),
    (
        torch.ones((100, 4, 120)),
        BaseTransformerLayer(
            attn=[MultiheadAttention(120, 5), MultiheadAttention(120, 5)],
            ffn=ReluDropIdentityMLP(120, 1024, 2),
            norm=torch.nn.LayerNorm(normalized_shape=120),
            operation_order=(
                "ffn",
                "norm",
                "self_attn",
                "norm",
                "cross_attn",
                "norm",
                "ffn",
                "norm",
                "ffn",
                "norm",
            ),
        ),
    ),
    (
        torch.ones((100, 4, 120)),
        BaseTransformerLayer(
            attn=[
                MultiheadAttention(120, 5),
                MultiheadAttention(120, 5),
                MultiheadAttention(120, 5),
                MultiheadAttention(120, 5),
                MultiheadAttention(120, 5),
                MultiheadAttention(120, 5),
            ],
            ffn=ReluDropIdentityMLP(120, 1024, 2),
            norm=torch.nn.LayerNorm(normalized_shape=120),
            operation_order=(
                "self_attn",
                "cross_attn",
                "self_attn",
                "cross_attn",
                "self_attn",
                "cross_attn",
            ),
        ),
    ),
]


@pytest.mark.parametrize("input, base_layer", TEST_CASES_BASE_LAYER)
def test_transformer_base_layer(input, base_layer):
    assert len(base_layer.attentions) == base_layer.operation_order.count(
        "self_attn"
    ) + base_layer.operation_order.count("cross_attn")
    assert len(base_layer.norms) == base_layer.operation_order.count("norm")
    assert len(base_layer.ffns) == base_layer.operation_order.count("ffn")

    key = torch.zeros_like(input)
    value = torch.zeros_like(input)
    query_pos = torch.zeros_like(input)
    key_pos = torch.zeros_like(input)
    out = base_layer(
        query=input,
        key=key,
        value=value,
        query_pos=query_pos,
        key_pos=key_pos,
    )
    assert input.shape == out.shape


## Functional Tests
EMBED_DIM = 128
N_HEADS = 8
FFN_DIM = 256
N_FEATURES = 4 * 4 * 4
BS = 2


class MockZeroAttention(MultiheadAttention):
    def __init__(self, *args, mock=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mock = mock
        assert self.mock is not None

    def forward(self, *args, **kwargs):
        assert self.mock is not None
        self.mock(*args, **kwargs)

        return torch.zeros_like(kwargs["query"])


class MockZeroFFN(ReluDropIdentityMLP):
    def __init__(self, *args, mock=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mock = mock
        assert self.mock is not None

    def forward(self, *args, **kwargs):
        assert self.mock is not None
        self.mock(*args, **kwargs)

        return torch.zeros_like(args[0])


def test_transformer_base_layer_self_attn_detr_enc():
    attention = MockZeroAttention(EMBED_DIM, N_HEADS, mock=Mock())
    base_layer = BaseTransformerLayer(
        attn=attention,
        ffn=None,
        norm=None,
        operation_order=("self_attn",),
    )

    mock = base_layer.attentions[0].mock

    input_query = torch.ones((N_FEATURES, BS, EMBED_DIM))  # [4*4*4, bs=2, embed_dim]
    input_query_pos = torch.zeros((N_FEATURES, BS, EMBED_DIM)).fill_(2)

    output = base_layer(query=input_query, query_pos=input_query_pos)

    assert mock.call_count == 1
    assert torch.allclose(mock.call_args[1]["query"], input_query)
    assert torch.allclose(mock.call_args[1]["key"], input_query)
    assert torch.allclose(mock.call_args[1]["value"], input_query)
    assert torch.allclose(mock.call_args[1]["identity"], input_query)
    assert torch.allclose(mock.call_args[1]["query_pos"], input_query_pos)
    assert torch.allclose(mock.call_args[1]["key_pos"], input_query_pos)


def test_transformer_base_layer_self_attn_detr_dec():
    attention = MockZeroAttention(EMBED_DIM, N_HEADS, mock=Mock())
    base_layer = BaseTransformerLayer(
        attn=attention,
        ffn=None,
        norm=None,
        operation_order=("self_attn",),
    )

    mock = base_layer.attentions[0].mock

    # output decoder (zero inint in implementation)
    target = torch.ones((N_FEATURES, BS, EMBED_DIM))  # [4*4*4, bs=2, embed_dim]
    # enc output
    memory = torch.zeros((N_FEATURES, BS, EMBED_DIM)).fill_(2)  # [4*4*4, bs=2, embed_dim]
    pos_embed = torch.zeros((N_FEATURES, BS, EMBED_DIM)).fill_(4)
    # object queries
    query_embed = torch.zeros((N_FEATURES, BS, EMBED_DIM)).fill_(3)

    output = base_layer(
        query=target,
        key=memory,
        value=memory,
        query_pos=query_embed,
        key_pos=pos_embed,
    )

    assert mock.call_count == 1
    assert torch.allclose(mock.call_args[1]["query"], target)
    assert torch.allclose(mock.call_args[1]["key"], target)
    assert torch.allclose(mock.call_args[1]["value"], target)
    assert torch.allclose(mock.call_args[1]["identity"], target)
    assert torch.allclose(mock.call_args[1]["query_pos"], query_embed)
    assert torch.allclose(mock.call_args[1]["key_pos"], query_embed)


def test_transformer_base_layer_cross_attn_detr_dec():
    attention = MockZeroAttention(EMBED_DIM, N_HEADS, mock=Mock())
    base_layer = BaseTransformerLayer(
        attn=attention,
        ffn=None,
        norm=None,
        operation_order=("cross_attn",),
    )

    mock = base_layer.attentions[0].mock

    # output decoder (zero inint in implementation)
    target = torch.ones((N_FEATURES, BS, EMBED_DIM))  # [4*4*4, bs=2, embed_dim]
    # enc output
    memory = torch.zeros((N_FEATURES, BS, EMBED_DIM)).fill_(2)  # [4*4*4, bs=2, embed_dim]
    pos_embed = torch.zeros((N_FEATURES, BS, EMBED_DIM)).fill_(4)
    # object queries
    query_embed = torch.zeros((N_FEATURES, BS, EMBED_DIM)).fill_(3)

    output = base_layer(
        query=target,
        key=memory,
        value=memory,
        query_pos=query_embed,
        key_pos=pos_embed,
    )

    assert mock.call_count == 1
    assert torch.allclose(mock.call_args[1]["query"], target)
    assert torch.allclose(mock.call_args[1]["key"], memory)
    assert torch.allclose(mock.call_args[1]["value"], memory)
    assert torch.allclose(mock.call_args[1]["identity"], target)
    assert torch.allclose(mock.call_args[1]["query_pos"], query_embed)
    assert torch.allclose(mock.call_args[1]["key_pos"], pos_embed)


def test_transformer_base_layer_classic_block_enc():
    attention = MockZeroAttention(EMBED_DIM, N_HEADS, mock=Mock())
    norm = torch.nn.LayerNorm(normalized_shape=EMBED_DIM)
    ffn = MockZeroFFN(EMBED_DIM, FFN_DIM, ffn_drop=0, num_layers=2, mock=Mock())
    base_layer = BaseTransformerLayer(
        attn=attention,
        ffn=ffn,
        norm=norm,
        operation_order=("self_attn", "norm", "ffn", "norm"),
    )

    input_query = torch.ones((N_FEATURES, BS, EMBED_DIM))  # [4*4*4, bs=2, embed_dim]
    input_query_pos = torch.zeros((N_FEATURES, BS, EMBED_DIM)).fill_(2)
    zeros_tensor = torch.zeros_like(input_query)
    output = base_layer(query=input_query, query_pos=input_query_pos)

    attn_mock = base_layer.attentions[0].mock
    ffn_mock = base_layer.ffns[0].mock

    assert attn_mock.call_count == 1
    assert torch.allclose(attn_mock.call_args[1]["identity"], input_query)

    assert ffn_mock.call_count == 1
    assert torch.allclose(ffn_mock.call_args[0][0], zeros_tensor)
    assert torch.allclose(ffn_mock.call_args[1]["identity"], zeros_tensor)


def test_transformer_base_layer_classic_block_dec():
    attention = MockZeroAttention(EMBED_DIM, N_HEADS, mock=Mock())
    norm = torch.nn.LayerNorm(normalized_shape=EMBED_DIM)
    ffn = MockZeroFFN(EMBED_DIM, FFN_DIM, ffn_drop=0, num_layers=2, mock=Mock())
    base_layer = BaseTransformerLayer(
        attn=attention,
        ffn=ffn,
        norm=norm,
        operation_order=(
            "self_attn",
            "norm",
            "cross_attn",
            "norm",
            "ffn",
            "norm",
        ),
    )

    # output decoder (zero inint in implementation)
    target = torch.ones((N_FEATURES, BS, EMBED_DIM))  # [4*4*4, bs=2, embed_dim]
    # enc output
    memory = torch.zeros((N_FEATURES, BS, EMBED_DIM)).fill_(2)  # [4*4*4, bs=2, embed_dim]
    pos_embed = torch.zeros((N_FEATURES, BS, EMBED_DIM)).fill_(4)
    # object queries
    query_embed = torch.zeros((N_FEATURES, BS, EMBED_DIM)).fill_(3)
    zeros_tensor = torch.zeros_like(memory)

    output = base_layer(
        query=target,
        key=memory,
        value=memory,
        query_pos=query_embed,
        key_pos=pos_embed,
    )

    attn_mock0 = base_layer.attentions[0].mock
    attn_mock1 = base_layer.attentions[1].mock
    ffn_mock = base_layer.ffns[0].mock

    assert attn_mock0.call_count == 1
    assert torch.allclose(attn_mock0.call_args[1]["identity"], target)

    assert attn_mock1.call_count == 1
    assert torch.allclose(attn_mock1.call_args[1]["identity"], zeros_tensor)

    assert ffn_mock.call_count == 1
    assert torch.allclose(ffn_mock.call_args[0][0], zeros_tensor)
    assert torch.allclose(ffn_mock.call_args[1]["identity"], zeros_tensor)
