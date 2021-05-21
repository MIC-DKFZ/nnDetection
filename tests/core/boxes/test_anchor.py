import pytest
from typing import Sequence

import torch

from nndet.core.boxes import get_anchor_generator

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.image_list import ImageList


@pytest.fixture
def generator2d():
    return get_anchor_generator(2)(sizes=(128,), aspect_ratios=(1.,))


@pytest.fixture
def generator3d():
    return get_anchor_generator(3)(sizes=(128,), aspect_ratios=(1.,), zsizes=(4, 8))


@pytest.fixture
def generator2ds():
    return get_anchor_generator(2, s_param=True)(width=(128,), height=(128,))


@pytest.fixture
def generator3ds():
    return get_anchor_generator(3, s_param=True)(width=(128,), height=(128,), depth=[(4, 8)])


def create_imagelist(img_size: Sequence[int]) -> ImageList:
    """
    img size includes batch and channel dimensions
    """
    return ImageList(torch.rand(img_size), [tuple(img_size[2:])])


def create_feature_maps_2d(img_size: Sequence[int],
                           fm_strides: Sequence[int]) -> Sequence[torch.Tensor]:
    """
    img size includes batch and channel dimensions
    """
    fm_sizes = [(img_size[0], int(img_size[1] * fms),
                 int(img_size[2] / fms), int(img_size[3] / fms)) for fms in fm_strides]
    return [torch.rand(size) for idx, size in enumerate(fm_sizes)]


def create_feature_maps_3d(img_size: Sequence[int],
                           fm_strides: Sequence[int]) -> Sequence[torch.Tensor]:
    """
    img size includes batch and channel dimensions
    """
    fm_sizes = [(img_size[0], int(img_size[1] * fms), int(img_size[2] / fms),
                 int(img_size[3] / fms), int(img_size[4] / fms)) for fms in fm_strides]
    return [torch.rand(size) for idx, size in enumerate(fm_sizes)]


def test_anchor_generator_2d(generator2d):
    img_size = [1, 3, 1, 1]
    fm_strides = [1]
    anchors = generator2d(torch.rand(img_size), create_feature_maps_2d(img_size, fm_strides))[0]
    assert all([a == b for a, b in zip(anchors.shape, (1, 4))])
    assert ((anchors == torch.tensor([-64., -64., 64., 64.])).all())

    anchors = generator2d(torch.rand(img_size), create_feature_maps_2d(img_size, fm_strides))[0]
    assert all([a == b for a, b in zip(anchors.shape, (1, 4))])


def test_anchor_generator_2d_torchvision():
    img_size = [1, 3, 16, 16]
    fm_strides = [2, 4, 8]
    torchvision_generator = AnchorGenerator(sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0))
    own_generator = get_anchor_generator(2)(sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0))

    anchors_vision = torchvision_generator(
        create_imagelist(img_size),
        create_feature_maps_2d(img_size, fm_strides),
    )
    anchors_own = own_generator(
        torch.rand(img_size),
        create_feature_maps_2d(img_size, fm_strides),
    )
    for vision, own in zip(anchors_vision, anchors_own):
        assert ((vision == own).all())


def test_anchor_generator_2d_assertion(generator2d):
    img_size = [1, 3, 2, 2]
    fm_strides = [1, 2]
    with pytest.raises(AssertionError):
        anchors = generator2d(
            torch.rand(img_size),
            create_feature_maps_2d(img_size, fm_strides),
        )


def test_anchor_generator_3d(generator3d):
    img_size = [1, 3, 1, 1, 1]
    fm_strides = [1]
    anchors = generator3d(
        torch.rand(img_size),
        create_feature_maps_3d(img_size, fm_strides),
    )[0]

    assert all([a == b for a, b in zip(anchors.shape, (2, 6))])
    assert ((anchors[0] == torch.tensor([-64., -64., 64., 64., -2., 2.])).all())
    assert ((anchors[1] == torch.tensor([-64., -64., 64., 64., -4., 4.])).all())

    anchors = generator3d(
        torch.rand(img_size),
        create_feature_maps_3d(img_size, fm_strides),
    )[0]
    assert all([a == b for a, b in zip(anchors.shape, (2, 6))])


def test_anchor_generator_3d_assertion(generator3d):
    img_size = [1, 3, 2, 2, 2]
    fm_strides = [1, 2]
    with pytest.raises(AssertionError):
        anchors = generator3d(
            torch.rand(img_size),
            create_feature_maps_3d(img_size, fm_strides),
        )


def test_anchor_generator_2ds(generator2ds):
    img_size = [1, 3, 1, 1]
    fm_strides = [1]
    anchors = generator2ds(
        torch.rand(img_size),
        create_feature_maps_2d(img_size, fm_strides),
    )[0]
    assert all([a == b for a, b in zip(anchors.shape, (1, 4))])
    assert ((anchors == torch.tensor([-64., -64., 64., 64.])).all())

    anchors = generator2ds(
        torch.rand(img_size),
        create_feature_maps_2d(img_size, fm_strides),
    )[0]
    assert all([a == b for a, b in zip(anchors.shape, (1, 4))])


def test_anchor_generator_3ds(generator3ds):
    img_size = [1, 3, 1, 1, 1]
    fm_strides = [1]
    anchors = generator3ds(
        torch.rand(img_size),
        create_feature_maps_3d(img_size, fm_strides),
    )[0]
    assert all([a == b for a, b in zip(anchors.shape, (2, 6))])
    assert ((anchors[0] == torch.tensor([-64., -64., 64., 64., -2., 2.])).all())
    assert ((anchors[1] == torch.tensor([-64., -64., 64., 64., -4., 4.])).all())

    anchors = generator3ds(
        torch.rand(img_size),
        create_feature_maps_3d(img_size, fm_strides),
    )[0]
    assert all([a == b for a, b in zip(anchors.shape, (2, 6))])
