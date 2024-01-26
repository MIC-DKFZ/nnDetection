import pytest
import torch

from nndet.nn.transformer.deformable_transformer import DeformableDETRTransformer
from nndet.nn.transformer.layers.deformable_detr import (
    DeformableDETRTransformerDecoder,
    DeformableDETRTransformerEncoder,
)


@pytest.fixture
def deformable_transformer():
    encoder = DeformableDETRTransformerEncoder()
    decoder = DeformableDETRTransformerDecoder()

    return DeformableDETRTransformer(
        encoder=encoder,
        decoder=decoder,
    )


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
