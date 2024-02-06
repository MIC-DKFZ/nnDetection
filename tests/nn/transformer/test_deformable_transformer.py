import math

import numpy as np
import pytest
import torch

from nndet.nn.heads.classifier.ffn import FocalFFNClassifier
from nndet.nn.heads.regressor.ffn import L1FFNRegressor
from nndet.nn.layers.linear import LayerLinearReluDrop
from nndet.nn.transformer.attention.multi_scale_deform_attn_3d import ms_deform_import
from nndet.nn.transformer.deformable_transformer import DeformableDETRTransformer
from nndet.nn.transformer.layers.deformable_detr import (
    DeformableDETRTransformerDecoder,
    DeformableDETRTransformerEncoder,
)

EMBED_DIM = 16


class IdentL1Regressor(L1FFNRegressor):
    _box_norm_fn = torch.nn.Identity()
    _inverse_box_norm_fn = torch.nn.Identity()


@pytest.fixture
def deformable_transformer():
    encoder = DeformableDETRTransformerEncoder(embed_dim=EMBED_DIM)
    decoder = DeformableDETRTransformerDecoder(embed_dim=EMBED_DIM, num_layers=3)
    classifier = FocalFFNClassifier(
        linear=LayerLinearReluDrop,
        in_channels=EMBED_DIM,
        internal_channels=EMBED_DIM,
        num_classes=2,
    )
    regressor = L1FFNRegressor(
        linear=LayerLinearReluDrop,
        in_channels=EMBED_DIM,
        internal_channels=EMBED_DIM,
        dim=3,
    )
    return DeformableDETRTransformer(
        encoder=encoder,
        decoder=decoder,
        classifier=classifier,
        regressor=regressor,
        two_stage=False,
        num_feature_levels=4,
    )


@pytest.fixture
def deformable_transformer_two_stage():
    encoder = DeformableDETRTransformerEncoder(embed_dim=EMBED_DIM)
    decoder = DeformableDETRTransformerDecoder(embed_dim=EMBED_DIM, num_layers=3)
    classifier = FocalFFNClassifier(
        linear=LayerLinearReluDrop,
        in_channels=EMBED_DIM,
        internal_channels=EMBED_DIM,
        num_classes=2,
        share_mlp=False,
        use_encoder_mlp=True,
    )
    regressor = L1FFNRegressor(
        linear=LayerLinearReluDrop,
        in_channels=EMBED_DIM,
        internal_channels=EMBED_DIM,
        dim=3,
        share_mlp=False,
        use_encoder_mlp=True,
    )
    return DeformableDETRTransformer(
        encoder=encoder,
        decoder=decoder,
        classifier=classifier,
        regressor=regressor,
        two_stage=True,
        two_stage_base_object_scale=0.05,
        num_feature_levels=4,
        two_stage_num_proposals=24,
    )


def _assert_module(module, two_stage: bool):
    features = [
        torch.ones(1, EMBED_DIM, 32, 32, 32),
        torch.ones(1, EMBED_DIM, 16, 16, 16),
        torch.ones(1, EMBED_DIM, 8, 8, 8),
        torch.ones(1, EMBED_DIM, 4, 4, 4),
    ]
    if two_stage:
        query_embed = None
    else:
        query_embed = torch.zeros(24, EMBED_DIM * 2)

    pos_embed = [torch.zeros_like(f) for f in features]

    inter_states, reference_out, enc_outputs = module(
        features=features,
        query_embed=query_embed,
        pos_embed=pos_embed,
    )

    assert inter_states.shape == (3, 1, 24, EMBED_DIM)
    dim = 6 if two_stage else 3
    assert reference_out.shape == (4, 1, 24, dim)  # 1 (initial) + 3 (intermediate)

    if two_stage:
        enc_outputs_class, enc_outputs_coord_unact = enc_outputs
        assert enc_outputs_class.shape == (1, 32**3 + 16**3 + 8**3 + 4**3, 2)
        assert enc_outputs_coord_unact.shape == (1, 32**3 + 16**3 + 8**3 + 4**3, 6)


def test_forward_two_stage_shape(deformable_transformer_two_stage):
    _assert_module(deformable_transformer_two_stage, two_stage=True)


def test_forward_shape(deformable_transformer):
    _assert_module(deformable_transformer, two_stage=False)


def test_get_reference_points(deformable_transformer):
    batch_size = 1
    spatial_shapes = torch.tensor([[2, 3, 4]])

    reference_points = deformable_transformer.get_reference_points(
        spatial_shapes=spatial_shapes,
        batch_size=batch_size,
        device=spatial_shapes.device,
    )

    expected_reference_points = torch.tensor(
        [
            [0.5 / 2, 0.5 / 3, 0.5 / 4],
            [0.5 / 2, 0.5 / 3, 1.5 / 4],
            [0.5 / 2, 0.5 / 3, 2.5 / 4],
            [0.5 / 2, 0.5 / 3, 3.5 / 4],
            [0.5 / 2, 1.5 / 3, 0.5 / 4],
            [0.5 / 2, 1.5 / 3, 1.5 / 4],
            [0.5 / 2, 1.5 / 3, 2.5 / 4],
            [0.5 / 2, 1.5 / 3, 3.5 / 4],
            [0.5 / 2, 2.5 / 3, 0.5 / 4],
            [0.5 / 2, 2.5 / 3, 1.5 / 4],
            [0.5 / 2, 2.5 / 3, 2.5 / 4],
            [0.5 / 2, 2.5 / 3, 3.5 / 4],
            [1.5 / 2, 0.5 / 3, 0.5 / 4],
            [1.5 / 2, 0.5 / 3, 1.5 / 4],
            [1.5 / 2, 0.5 / 3, 2.5 / 4],
            [1.5 / 2, 0.5 / 3, 3.5 / 4],
            [1.5 / 2, 1.5 / 3, 0.5 / 4],
            [1.5 / 2, 1.5 / 3, 1.5 / 4],
            [1.5 / 2, 1.5 / 3, 2.5 / 4],
            [1.5 / 2, 1.5 / 3, 3.5 / 4],
            [1.5 / 2, 2.5 / 3, 0.5 / 4],
            [1.5 / 2, 2.5 / 3, 1.5 / 4],
            [1.5 / 2, 2.5 / 3, 2.5 / 4],
            [1.5 / 2, 2.5 / 3, 3.5 / 4],
        ]
    )[None, :, None]
    assert torch.allclose(reference_points, expected_reference_points)


def test_get_reference_points_shape(deformable_transformer):
    batch_size = 2
    spatial_shapes = torch.tensor([[2, 3, 4], [5, 6, 7]])

    reference_points = deformable_transformer.get_reference_points(
        spatial_shapes=spatial_shapes,
        batch_size=batch_size,
        device=spatial_shapes.device,
    )
    expected_shape = (2, 2 * 3 * 4 + 5 * 6 * 7, 2, 3)
    assert tuple(reference_points.shape) == expected_shape


def test_gen_encoder_output_proposals(deformable_transformer_two_stage):
    spatial_shapes = [[2, 3, 4]]
    num_elements = 2 * 3 * 4

    memory = torch.zeros(1, num_elements, EMBED_DIM)  # B, ref_points, embed_dim
    spatial_shapes = torch.tensor(spatial_shapes)

    output_memory, output_proposals = deformable_transformer_two_stage.gen_encoder_output_proposals(
        memory=memory,
        spatial_shapes=spatial_shapes,
    )

    assert output_memory.shape == (1, num_elements, EMBED_DIM)
    assert output_proposals.shape == (1, num_elements, 6)

    elem = 0.05
    output_proposals_expected = torch.tensor(
        [
            [0.5 / 2, 0.5 / 3, 0.5 / 4, elem, elem, elem],
            [0.5 / 2, 0.5 / 3, 1.5 / 4, elem, elem, elem],
            [0.5 / 2, 0.5 / 3, 2.5 / 4, elem, elem, elem],
            [0.5 / 2, 0.5 / 3, 3.5 / 4, elem, elem, elem],
            [0.5 / 2, 1.5 / 3, 0.5 / 4, elem, elem, elem],
            [0.5 / 2, 1.5 / 3, 1.5 / 4, elem, elem, elem],
            [0.5 / 2, 1.5 / 3, 2.5 / 4, elem, elem, elem],
            [0.5 / 2, 1.5 / 3, 3.5 / 4, elem, elem, elem],
            [0.5 / 2, 2.5 / 3, 0.5 / 4, elem, elem, elem],
            [0.5 / 2, 2.5 / 3, 1.5 / 4, elem, elem, elem],
            [0.5 / 2, 2.5 / 3, 2.5 / 4, elem, elem, elem],
            [0.5 / 2, 2.5 / 3, 3.5 / 4, elem, elem, elem],
            [1.5 / 2, 0.5 / 3, 0.5 / 4, elem, elem, elem],
            [1.5 / 2, 0.5 / 3, 1.5 / 4, elem, elem, elem],
            [1.5 / 2, 0.5 / 3, 2.5 / 4, elem, elem, elem],
            [1.5 / 2, 0.5 / 3, 3.5 / 4, elem, elem, elem],
            [1.5 / 2, 1.5 / 3, 0.5 / 4, elem, elem, elem],
            [1.5 / 2, 1.5 / 3, 1.5 / 4, elem, elem, elem],
            [1.5 / 2, 1.5 / 3, 2.5 / 4, elem, elem, elem],
            [1.5 / 2, 1.5 / 3, 3.5 / 4, elem, elem, elem],
            [1.5 / 2, 2.5 / 3, 0.5 / 4, elem, elem, elem],
            [1.5 / 2, 2.5 / 3, 1.5 / 4, elem, elem, elem],
            [1.5 / 2, 2.5 / 3, 2.5 / 4, elem, elem, elem],
            [1.5 / 2, 2.5 / 3, 3.5 / 4, elem, elem, elem],
        ]
    )[None]
    output_proposals_expected = deformable_transformer_two_stage.regressor.apply_inverse_non_lin(
        output_proposals_expected
    )
    assert torch.allclose(output_proposals, output_proposals_expected)


def test_gen_encoder_output_proposals_shape(deformable_transformer_two_stage):
    spatial_shapes = [[4, 6, 8], [16, 24, 32], [64, 64, 64]]
    num_elements = 4 * 6 * 8 + 16 * 24 * 32 + 64 * 64 * 64

    memory = torch.zeros(1, num_elements, EMBED_DIM)  # B, ref_points, embed_dim
    spatial_shapes = torch.tensor(spatial_shapes)

    output_memory, output_proposals = deformable_transformer_two_stage.gen_encoder_output_proposals(
        memory=memory,
        spatial_shapes=spatial_shapes,
    )

    assert output_memory.shape == (1, num_elements, EMBED_DIM)
    assert output_proposals.shape == (1, num_elements, 6)


def test_get_proposal_pos_embed():
    encoder = DeformableDETRTransformerEncoder(embed_dim=4, num_heads=1)
    decoder = DeformableDETRTransformerDecoder(embed_dim=4, num_heads=1)
    regressor = IdentL1Regressor(
        linear=LayerLinearReluDrop,
        in_channels=2,
        internal_channels=2,
        dim=3,
    )

    transformer = DeformableDETRTransformer(
        encoder=encoder,
        decoder=decoder,
        regressor=regressor,
        two_stage=True,
    )

    reg_coords = torch.arange(12).reshape(1, 2, 6).to(dtype=torch.double)
    pos_embed_coords = transformer.get_proposal_pos_embed(reg_coords)

    temp = 10000
    d = 2
    expected_pos_embed_coords = torch.tensor(
        [
            [
                # block 1
                torch.sin(2 * np.pi * 0 * 1 / (torch.tensor(temp, dtype=torch.double) ** (2 * 0 / d))),
                torch.cos(2 * np.pi * 0 * 1 / (torch.tensor(temp, dtype=torch.double) ** (2 * 0 / d))),
                torch.sin(2 * np.pi * 1 * 1 / (torch.tensor(temp, dtype=torch.double) ** (2 * 0 / d))),
                torch.cos(2 * np.pi * 1 * 1 / (torch.tensor(temp, dtype=torch.double) ** (2 * 0 / d))),
                torch.sin(2 * np.pi * 2 * 1 / (torch.tensor(temp, dtype=torch.double) ** (2 * 0 / d))),
                torch.cos(2 * np.pi * 2 * 1 / (torch.tensor(temp, dtype=torch.double) ** (2 * 0 / d))),
                torch.sin(2 * np.pi * 3 * 1 / (torch.tensor(temp, dtype=torch.double) ** (2 * 0 / d))),
                torch.cos(2 * np.pi * 3 * 1 / (torch.tensor(temp, dtype=torch.double) ** (2 * 0 / d))),
                torch.sin(2 * np.pi * 4 * 1 / (torch.tensor(temp, dtype=torch.double) ** (2 * 0 / d))),
                torch.cos(2 * np.pi * 4 * 1 / (torch.tensor(temp, dtype=torch.double) ** (2 * 0 / d))),
                torch.sin(2 * np.pi * 5 * 1 / (torch.tensor(temp, dtype=torch.double) ** (2 * 0 / d))),
                torch.cos(2 * np.pi * 5 * 1 / (torch.tensor(temp, dtype=torch.double) ** (2 * 0 / d))),
            ],
            [
                # block 2
                torch.sin(2 * np.pi * 6 * 1 / (torch.tensor(temp, dtype=torch.double) ** (2 * 0 / d))),
                torch.cos(2 * np.pi * 6 * 1 / (torch.tensor(temp, dtype=torch.double) ** (2 * 0 / d))),
                torch.sin(2 * np.pi * 7 * 1 / (torch.tensor(temp, dtype=torch.double) ** (2 * 0 / d))),
                torch.cos(2 * np.pi * 7 * 1 / (torch.tensor(temp, dtype=torch.double) ** (2 * 0 / d))),
                torch.sin(2 * np.pi * 8 * 1 / (torch.tensor(temp, dtype=torch.double) ** (2 * 0 / d))),
                torch.cos(2 * np.pi * 8 * 1 / (torch.tensor(temp, dtype=torch.double) ** (2 * 0 / d))),
                torch.sin(2 * np.pi * 9 * 1 / (torch.tensor(temp, dtype=torch.double) ** (2 * 0 / d))),
                torch.cos(2 * np.pi * 9 * 1 / (torch.tensor(temp, dtype=torch.double) ** (2 * 0 / d))),
                torch.sin(2 * np.pi * 10 * 1 / (torch.tensor(temp, dtype=torch.double) ** (2 * 0 / d))),
                torch.cos(2 * np.pi * 10 * 1 / (torch.tensor(temp, dtype=torch.double) ** (2 * 0 / d))),
                torch.sin(2 * np.pi * 11 * 1 / (torch.tensor(temp, dtype=torch.double) ** (2 * 0 / d))),
                torch.cos(2 * np.pi * 11 * 1 / (torch.tensor(temp, dtype=torch.double) ** (2 * 0 / d))),
            ],
        ]
    )

    assert pos_embed_coords.shape == (1, 2, 12)
    assert torch.allclose(pos_embed_coords, expected_pos_embed_coords)


TEST_CASES_SHAPE_DEFORMABLE = [
    (
        DeformableDETRTransformerEncoder(
            embed_dim=64,
            num_heads=2,
            num_layers=2,
            num_feature_levels=2,
        ),  # encoder
        DeformableDETRTransformerDecoder(
            embed_dim=64,
            num_heads=2,
            num_layers=2,
            num_feature_levels=2,
        ),  # decoder
        FocalFFNClassifier(
            linear=LayerLinearReluDrop,
            in_channels=64,
            internal_channels=32,
            num_classes=2,
            use_encoder_mlp=True,
            share_mlp=False,
        ),  # classifier
        L1FFNRegressor(
            linear=LayerLinearReluDrop,
            in_channels=64,
            internal_channels=32,
            dim=3,
            use_encoder_mlp=True,
            share_mlp=False,
        ),  # regressor
    ),
]


@pytest.mark.skipif(not ms_deform_import, reason="nnDetection was not build with GPU support")
@pytest.mark.parametrize(
    "encoder,decoder,classifier,regressor",
    TEST_CASES_SHAPE_DEFORMABLE,
)
@pytest.mark.parametrize("two_stage", [True, False])
def test_deformable_transformer_check_output_shape(
    encoder,
    decoder,
    classifier,
    regressor,
    two_stage: bool,
):
    torch.manual_seed(0)

    features = [torch.rand((4, 64, 16, 16, 16)), torch.rand((4, 64, 8, 8, 8))]
    query_embed = torch.rand(24, 2 * 64)  # n_det, embed_dim
    pos_embed = [torch.zeros((4, 64, 16, 16, 16)), torch.zeros((4, 64, 8, 8, 8))]

    transformer = DeformableDETRTransformer(
        encoder=encoder,
        decoder=decoder,
        classifier=classifier,
        regressor=regressor,
        num_feature_levels=len(features),
        two_stage=two_stage,
        two_stage_num_proposals=24,
    )
    inter_states, references, enc_outputs = transformer(
        features=features,
        query_embed=query_embed,
        pos_embed=pos_embed,
    )
