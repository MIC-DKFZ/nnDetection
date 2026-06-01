# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from nndet.nn.heads.regressor.ffn import L1FFNRegressor
from nndet.nn.layers.linear import LayerLinearReluDrop
from nndet.nn.transformer.attention.multi_scale_deform_attn import ms_deform_import
from nndet.nn.transformer.layers.deformable_detr import (
    DeformableDETRTransformerDecoder,
    DeformableDETRTransformerEncoder,
)
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


TEST_CASES_DEFORMABLE_ENCODER = [
    (
        DeformableDETRTransformerEncoder(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            feedforward_dim=FFN_DIM,
            dim=DIM,
            num_feature_levels=2,
            num_points=4,
        ),
        EMBED_DIM,  # embed_dim
        3,  # spatial dimensions
    ),
]


TEST_CASES_DEFORMABLE_DECODER = [
    (
        DeformableDETRTransformerDecoder(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            feedforward_dim=FFN_DIM,
            dim=DIM,
            num_feature_levels=2,
            num_points=4,
        ),
        EMBED_DIM,  # embed_dim
        3,  # spatial dimensions
    ),
    (
        DeformableDETRTransformerDecoder(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            feedforward_dim=FFN_DIM,
            dim=DIM,
            num_feature_levels=2,
            num_points=4,
            regressor=L1FFNRegressor(
                linear=LayerLinearReluDrop,
                in_channels=EMBED_DIM,
                internal_channels=EMBED_DIM,
                dim=DIM,
            ),
        ),
        EMBED_DIM,  # embed_dim
        3,  # spatial dimensions
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


@pytest.mark.parametrize(
    "module,embed_dim,dim",
    TEST_CASES_DEFORMABLE_ENCODER,
)
def test_deformable_detr_encoder_smoke(module, embed_dim: int, dim: int):
    torch.manual_seed(0)

    spatial_shapes = [(4, 8, 8), (8, 16, 16)]
    level_start_index = [0, 4 * 8 * 8]
    n_points = 4 * 8 * 8 + 8 * 16 * 16
    bs = 2

    query = torch.rand((bs, n_points, embed_dim))
    query_pos = torch.rand_like(query)
    spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=query.device)
    refs_cccddd_norm = torch.rand((bs, n_points, len(spatial_shapes), dim))

    memory = module(
        query=query,  # bs, level * p-dims, embed_dim
        key=None,
        value=None,
        query_pos=query_pos,  # bs, level * p-dims, embed_dim
        key_pos=None,
        spatial_shapes=spatial_shapes,
        refs_cccddd_norm=refs_cccddd_norm,  # bs, num_token, num_level, 2
        level_start_index=level_start_index,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
    )
    assert memory.shape == (bs, n_points, embed_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda gpu available")
@pytest.mark.skipif(not ms_deform_import, reason="nnDetection was not build with GPU support")
@pytest.mark.parametrize(
    "module,embed_dim,dim",
    TEST_CASES_DEFORMABLE_DECODER,
)
@pytest.mark.parametrize("two_stage", [True, False])
def test_deformable_detr_decoder_smoke(module, embed_dim: int, dim: int, two_stage: bool):
    torch.manual_seed(0)

    spatial_shapes = [(4, 8, 8), (8, 16, 16)]
    level_start_index = [0, 4 * 8 * 8]
    n_points = 4 * 8 * 8 + 8 * 16 * 16
    bs = 2
    n_pred = 24

    query = torch.rand((bs, n_pred, embed_dim))
    query_pos = torch.rand_like(query)
    memory = torch.rand((bs, n_points, embed_dim))
    spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=query.device)
    if two_stage:
        ref_dim = dim * 2
    else:
        ref_dim = dim
    refs_cccddd_norm = torch.rand((bs, n_pred, ref_dim))

    inter_states, inter_references = module(
        query=query,  # bs, num_queries, embed_dims; num_queries = topk
        key=None,  # bs, num_tokens, embed_dims
        value=memory,  # bs, num_tokens, embed_dims
        query_pos=query_pos,
        key_pos=query_pos,
        refs_cccddd_norm=refs_cccddd_norm,  # num_queries, 6
        spatial_shapes=spatial_shapes,  # nlvl, 2
        level_start_index=level_start_index,  # nlvl
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
    )

    num_layers = len(module.layers)
    assert inter_states.shape == (num_layers, bs, n_pred, embed_dim)
    output_dim = dim if module.regressor is None and two_stage == False else dim * 2
    assert inter_references.shape == (num_layers, bs, n_pred, output_dim)
