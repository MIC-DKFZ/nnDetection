from typing import Optional

import pytest
import torch

from nndet.nn.heads.regressor.ffn import L1FFNRegressor
from nndet.nn.layers.linear import LayerLinearReluDrop
from nndet.nn.transformer.attention.multi_scale_deform_attn import ms_deform_import
from nndet.nn.transformer.layers.deformable_detr import DeformableDETRTransformerDecoder

DIM = 3
N_PRED = 12
EMBED_DIM = 64
N_DECODER_LAYERS = 3


class AddOneModule(torch.nn.Module):
    def forward(self, x):
        return x + 1


class SeqRegressor(L1FFNRegressor):
    _box_norm_fn = AddOneModule()
    _inverse_box_norm_fn = AddOneModule()

    def forward(self, features: torch.Tensor, layer: Optional[int] = None):
        bs, num_p, _ = features.shape
        return torch.zeros((bs, num_p, 2 * self.dim), device=features.device, dtype=features.dtype).fill_(0.5)


@pytest.fixture
def deformable_detr_decoder():
    regressor = SeqRegressor(
        linear=LayerLinearReluDrop,
        in_channels=EMBED_DIM,
        internal_channels=EMBED_DIM,
        dim=DIM,
    )
    return DeformableDETRTransformerDecoder(
        embed_dim=EMBED_DIM,
        num_heads=1,
        num_layers=N_DECODER_LAYERS,
        return_intermediate=True,
        dim=DIM,
        num_feature_levels=3,
        regressor=regressor,
    )


@pytest.fixture
def deformable_detr_decoder():
    regressor = SeqRegressor(
        linear=LayerLinearReluDrop,
        in_channels=EMBED_DIM,
        internal_channels=EMBED_DIM,
        dim=DIM,
    )
    return DeformableDETRTransformerDecoder(
        embed_dim=EMBED_DIM,
        num_heads=1,
        num_layers=N_DECODER_LAYERS,
        return_intermediate=True,
        dim=DIM,
        num_feature_levels=3,
        regressor=regressor,
    )


@pytest.mark.skipif(not ms_deform_import, reason="nnDetection was not build with GPU support")
def test_deformable_detr_decoder_two_stage(deformable_detr_decoder):
    bs = 2
    spatial_shapes = [(4, 4, 4), (2, 2, 2), (1, 1, 1)]
    spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.int64)
    level_start_index = [0, 16, 20]
    n_points = 4 * 4 * 4 + 2 * 2 * 2 + 1 * 1 * 1

    query = torch.rand((bs, N_PRED, EMBED_DIM))
    value = torch.rand((bs, n_points, EMBED_DIM))
    query_pos = torch.zeros_like(query)
    refs_cccddd_norm = torch.zeros((bs, N_PRED, DIM * 2), dtype=query.dtype, device=query.device)

    output, reference_points = deformable_detr_decoder(
        query=query,
        key=None,
        value=value,
        query_pos=query_pos,
        key_pos=query_pos,
        refs_cccddd_norm=refs_cccddd_norm,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
    )

    expected_output_shape = (N_DECODER_LAYERS, bs, N_PRED, EMBED_DIM)
    assert tuple(output.shape) == expected_output_shape

    expected_reference_points_shape = (N_DECODER_LAYERS, bs, N_PRED, DIM * 2)
    assert tuple(reference_points.shape) == expected_reference_points_shape

    expected_reference_points = []
    for i in range(1, N_DECODER_LAYERS + 1):
        expected_reference_points.append(
            torch.zeros((bs, N_PRED, DIM * 2), dtype=query.dtype, device=query.device).fill_(i * 2.5)
        )
    expected_reference_points = torch.stack(expected_reference_points, dim=0)

    assert torch.allclose(reference_points, expected_reference_points)


@pytest.mark.skipif(not ms_deform_import, reason="nnDetection was not build with GPU support")
def test_deformable_detr_decoder(deformable_detr_decoder):
    bs = 2
    spatial_shapes = [(4, 4, 4), (2, 2, 2), (1, 1, 1)]
    spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.int64)
    level_start_index = [0, 16, 20]
    n_points = 4 * 4 * 4 + 2 * 2 * 2 + 1 * 1 * 1

    query = torch.rand((bs, N_PRED, EMBED_DIM))
    value = torch.rand((bs, n_points, EMBED_DIM))
    query_pos = torch.zeros_like(query)
    refs_cccddd_norm = torch.zeros((bs, N_PRED, DIM), dtype=query.dtype, device=query.device)

    output, reference_points = deformable_detr_decoder(
        query=query,
        key=None,
        value=value,
        query_pos=query_pos,
        key_pos=query_pos,
        refs_cccddd_norm=refs_cccddd_norm,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
    )

    expected_output_shape = (N_DECODER_LAYERS, bs, N_PRED, EMBED_DIM)
    assert tuple(output.shape) == expected_output_shape

    expected_reference_points_shape = (N_DECODER_LAYERS, bs, N_PRED, DIM * 2)
    assert tuple(reference_points.shape) == expected_reference_points_shape

    expected_reference_points = []
    for i in range(1, N_DECODER_LAYERS + 1):
        centers = torch.zeros((bs, N_PRED, DIM), dtype=query.dtype, device=query.device).fill_(i * 2.5)
        sizes = torch.zeros((bs, N_PRED, DIM), dtype=query.dtype, device=query.device).fill_(i * 2.5 - 1.0)
        expected_reference_points.append(torch.cat([centers, sizes], dim=-1))
    expected_reference_points = torch.stack(expected_reference_points, dim=0)

    assert torch.allclose(reference_points, expected_reference_points)
