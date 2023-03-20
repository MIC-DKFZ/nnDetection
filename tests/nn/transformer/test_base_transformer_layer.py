# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch.nn

from nndet.nn.layers.mlp import ReluDropIdentityMLP
from nndet.nn.transformer.attention.attention import MultiheadAttention
from nndet.nn.transformer.layers.base_layer import (
    BaseTransformerLayer,
    TransformerLayerSequence,
)

# test if all modules are accessed
# test if output shape is correct


def test_transformer_base_layer():
    return


TEST_CASES_LAYER_SEQUENCE = [
    (
        torch.ones((100, 4, 256)),
        BaseTransformerLayer(
            attn=MultiheadAttention(256, 8),
            ffn=ReluDropIdentityMLP(256, 1024, 2),
            norm=torch.nn.LayerNorm(normalized_shape=256),
            operation_order=("self_attn", "norm", "ffn", "norm"),
        ),
        6,
    ),
    (
        torch.ones((100, 4, 128)),
        [
            BaseTransformerLayer(
                attn=MultiheadAttention(128, 8),
                ffn=ReluDropIdentityMLP(128, 2048, 2),
                norm=torch.nn.LayerNorm(normalized_shape=128),
                operation_order=("self_attn", "norm", "ffn", "norm"),
            ),
            BaseTransformerLayer(
                attn=MultiheadAttention(128, 8),
                ffn=ReluDropIdentityMLP(128, 1024, 2),
                norm=torch.nn.LayerNorm(normalized_shape=128),
                operation_order=("ffn", "norm", "self_attn", "norm"),
            ),
        ],
        2,
    ),
]


@pytest.mark.parametrize("input, base_layer, num_layers", TEST_CASES_LAYER_SEQUENCE)
def test_transformer_layer_sequence(input, base_layer, num_layers):
    layer_sequence = TransformerLayerSequence(base_layer, num_layers)
    assert isinstance(layer_sequence.layers, torch.nn.ModuleList)
    assert layer_sequence.num_layers == num_layers
    assert len(layer_sequence.layers) == num_layers
    input_shape = input.shape
    for layer in layer_sequence.layers:
        input = layer(input)
    assert input.shape == input_shape
