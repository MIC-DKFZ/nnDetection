from typing import Sequence

import pytest
import torch

from nndet.core.boxes.anchors import AnchorGenerator2D, AnchorGenerator3D


@pytest.fixture
def generator2d():
    return AnchorGenerator2D(width=(64,), height=(128,))


@pytest.fixture
def generator3d():
    return AnchorGenerator3D(width=(64,), height=(128,), depth=[(4, 8)])


def create_feature_maps_2d(img_size: Sequence[int], fm_strides: Sequence[int]) -> Sequence[torch.Tensor]:
    """
    img size includes batch and channel dimensions
    """
    fm_sizes = [
        (
            img_size[0],
            int(img_size[1] * fms),
            int(img_size[2] / fms),
            int(img_size[3] / fms),
        )
        for fms in fm_strides
    ]
    return [torch.rand(size) for idx, size in enumerate(fm_sizes)]


def create_feature_maps_3d(img_size: Sequence[int], fm_strides: Sequence[int]) -> Sequence[torch.Tensor]:
    """
    img size includes batch and channel dimensions
    """
    fm_sizes = [
        (
            img_size[0],
            int(img_size[1] * fms),
            int(img_size[2] / fms),
            int(img_size[3] / fms),
            int(img_size[4] / fms),
        )
        for fms in fm_strides
    ]
    return [torch.rand(size) for idx, size in enumerate(fm_sizes)]


def test_anchor_generator_3d_assertion(generator3d):
    img_size = [1, 3, 2, 2, 2]
    fm_strides = [1, 2]
    with pytest.raises(AssertionError):
        anchors = generator3d(
            torch.rand(img_size),
            create_feature_maps_3d(img_size, fm_strides),
        )


def test_anchor_generator_2d(generator2d):
    img_size = [1, 3, 1, 1]
    fm_strides = [1]
    anchors = generator2d(
        torch.rand(img_size),
        create_feature_maps_2d(img_size, fm_strides),
    )[0]
    assert all([a == b for a, b in zip(anchors.shape, (1, 4))])
    assert (anchors == torch.tensor([-32.0, -64.0, 32.0, 64.0])).all()

    anchors = generator2d(
        torch.rand(img_size),
        create_feature_maps_2d(img_size, fm_strides),
    )[0]
    assert all([a == b for a, b in zip(anchors.shape, (1, 4))])


def test_anchor_generator_3d(generator3d):
    img_size = [1, 3, 1, 1, 1]
    fm_strides = [1]
    anchors = generator3d(
        torch.rand(img_size),
        create_feature_maps_3d(img_size, fm_strides),
    )[0]
    assert all([a == b for a, b in zip(anchors.shape, (2, 6))])
    assert (anchors[0] == torch.tensor([-32.0, -64.0, 32.0, 64.0, -2.0, 2.0])).all()
    assert (anchors[1] == torch.tensor([-32.0, -64.0, 32.0, 64.0, -4.0, 4.0])).all()

    anchors = generator3d(
        torch.rand(img_size),
        create_feature_maps_3d(img_size, fm_strides),
    )[0]
    assert all([a == b for a, b in zip(anchors.shape, (2, 6))])


def test_anchor_generator_3d_order_ax0(generator3d):
    img_size = [1, 3, 4, 2, 2]
    fm_strides = [2]
    anchors = generator3d(
        torch.rand(img_size),
        create_feature_maps_3d(img_size, fm_strides),
    )[0]
    assert all([a == b for a, b in zip(anchors.shape, (4, 6))])
    assert (anchors[0] == torch.tensor([-32.0, -64.0, 32.0, 64.0, -2.0, 2.0])).all()
    assert (anchors[1] == torch.tensor([-32.0, -64.0, 32.0, 64.0, -4.0, 4.0])).all()
    assert (anchors[2] == torch.tensor([-30.0, -64.0, 34.0, 64.0, -2.0, 2.0])).all()
    assert (anchors[3] == torch.tensor([-30.0, -64.0, 34.0, 64.0, -4.0, 4.0])).all()


def test_anchor_generator_3d_order_ax1(generator3d):
    img_size = [1, 3, 2, 4, 2]
    fm_strides = [2]
    anchors = generator3d(
        torch.rand(img_size),
        create_feature_maps_3d(img_size, fm_strides),
    )[0]
    assert all([a == b for a, b in zip(anchors.shape, (4, 6))])
    assert (anchors[0] == torch.tensor([-32.0, -64.0, 32.0, 64.0, -2.0, 2.0])).all()
    assert (anchors[1] == torch.tensor([-32.0, -64.0, 32.0, 64.0, -4.0, 4.0])).all()
    assert (anchors[2] == torch.tensor([-32.0, -62.0, 32.0, 66.0, -2.0, 2.0])).all()
    assert (anchors[3] == torch.tensor([-32.0, -62.0, 32.0, 66.0, -4.0, 4.0])).all()


def test_anchor_generator_3d_order_ax2(generator3d):
    img_size = [1, 3, 2, 2, 4]
    fm_strides = [2]
    anchors = generator3d(
        torch.rand(img_size),
        create_feature_maps_3d(img_size, fm_strides),
    )[0]
    assert all([a == b for a, b in zip(anchors.shape, (4, 6))])
    assert (anchors[0] == torch.tensor([-32.0, -64.0, 32.0, 64.0, -2.0, 2.0])).all()
    assert (anchors[1] == torch.tensor([-32.0, -64.0, 32.0, 64.0, -4.0, 4.0])).all()
    assert (anchors[2] == torch.tensor([-32.0, -64.0, 32.0, 64.0, -0.0, 4.0])).all()
    assert (anchors[3] == torch.tensor([-32.0, -64.0, 32.0, 64.0, -2.0, 6.0])).all()
