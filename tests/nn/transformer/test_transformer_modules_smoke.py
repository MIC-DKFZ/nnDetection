# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from nndet.nn.transformer.layers.detr import (
    DETRTransformerDecoder,
    DETRTransformerEncoder,
)

DIM = 3
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 3
FFN_DIM = 128

NUM_FEATURES = 128
BS = 2

TEST_CASES_ENCODER = [
    (
        DETRTransformerEncoder(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            feedforward_dim=FFN_DIM,
            dim=DIM,
        ),
        (NUM_FEATURES, BS, EMBED_DIM),  # qkv_shape
        (NUM_FEATURES, BS, EMBED_DIM),  # expected_output_shape0
    ),
]


TEST_CASES_DECODER = [
    (
        DETRTransformerDecoder(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            feedforward_dim=FFN_DIM,
            dim=DIM,
            return_intermediate=False,
            post_norm=False,
        ),
        (8, BS, EMBED_DIM),  # q_shape [num_query, bs, c]
        (NUM_FEATURES, BS, EMBED_DIM),  # kv_shape
        (1, 8, BS, EMBED_DIM),  # expected_output_shape0
        None,  # expected_output_shape1
    ),
    (
        DETRTransformerDecoder(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            feedforward_dim=FFN_DIM,
            dim=DIM,
            return_intermediate=False,
            post_norm=True,
        ),
        (8, BS, EMBED_DIM),  # q_shape [num_query, bs, c]
        (NUM_FEATURES, BS, EMBED_DIM),  # kv_shape
        (1, 8, BS, EMBED_DIM),  # expected_output_shape0
        None,  # expected_output_shape1
    ),
    (
        DETRTransformerDecoder(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            feedforward_dim=FFN_DIM,
            dim=DIM,
            return_intermediate=True,
            post_norm=False,
        ),
        (8, BS, EMBED_DIM),  # q_shape
        (NUM_FEATURES, BS, EMBED_DIM),  # kv_shape
        (NUM_LAYERS, 8, BS, EMBED_DIM),  # expected_output_shape0
        None,  # expected_output_shape1
    ),
    (
        DETRTransformerDecoder(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            feedforward_dim=FFN_DIM,
            dim=DIM,
            return_intermediate=True,
            post_norm=True,
        ),
        (8, BS, EMBED_DIM),  # q_shape
        (NUM_FEATURES, BS, EMBED_DIM),  # kv_shape
        (NUM_LAYERS, 8, BS, EMBED_DIM),  # expected_output_shape0
        None,  # expected_output_shape1
    ),
]


@pytest.mark.parametrize("module,qkv_shape,expected_output_shape0", TEST_CASES_ENCODER)
def test_detr_encoder(
    module,
    qkv_shape,
    expected_output_shape0,
):
    torch.manual_seed(0)
    query = torch.rand(qkv_shape)
    query_pos = torch.rand(qkv_shape)

    output0 = module(query=query, query_pos=query_pos)
    assert tuple(output0.shape) == expected_output_shape0


@pytest.mark.parametrize(
    "module,q_shape,kv_shape,expected_output_shape0,expected_output_shape1",
    TEST_CASES_DECODER,
)
def test_detr_decoder(
    module,
    q_shape,
    kv_shape,
    expected_output_shape0,
    expected_output_shape1,
):
    torch.manual_seed(0)
    query = torch.rand(q_shape)
    key = torch.rand(kv_shape)
    value = torch.rand(kv_shape)
    query_pos = torch.rand(q_shape)
    key_pos = torch.zeros(kv_shape)

    output0, output1 = module(
        query=query,
        key=key,
        value=value,
        query_pos=query_pos,
        key_pos=key_pos,
    )

    assert tuple(output0.shape) == expected_output_shape0
    if expected_output_shape1 is None:
        assert output1 is None
    else:
        assert tuple(output1.shape) == expected_output_shape1
