# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

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
            operation_order=("ffn", "norm", "self_attn", "norm", "cross_attn", "norm", "ffn", "norm", "ffn", "norm"),
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
            operation_order=("self_attn", "cross_attn", "self_attn", "cross_attn", "self_attn", "cross_attn"),
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

    out = base_layer(input)
    assert input.shape == out.shape
