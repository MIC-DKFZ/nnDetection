import math

import pytest
import torch

from nndet.core.boxes import *
from nndet.core.boxes.ops import cat_and_index, distance_box_iou_3d_paired


@pytest.fixture
def boxes0_2d():
    return torch.tensor([[0, 0, 2, 2], [0, 0, 2, 2]]).float()


@pytest.fixture
def boxes1_2d():
    return torch.tensor([[1, 1, 3, 3], [1, 1, 3, 3], [1, 1, 3, 3]]).float()


@pytest.fixture
def boxes0_3d():
    return torch.tensor([[0, 0, 2, 2, 0, 2], [0, 0, 2, 2, 0, 2]]).float()


@pytest.fixture
def boxes1_3d():
    return torch.tensor(
        [[1, 1, 3, 3, 1, 3], [1, 1, 3, 3, 1, 3], [1, 1, 3, 3, 1, 3]]
    ).float()


def check_ious(boxes: torch.Tensor, similarity_fn):
    """
    Computes iou from boxes to boxes -> should always result in 1
    """


def test_box_area_2d(boxes0_2d, boxes1_2d):
    areas0 = box_area(boxes0_2d)
    areas1 = box_area(boxes1_2d)

    assert (areas0 == torch.tensor([4, 4])).all()
    assert (areas1 == torch.tensor([4, 4, 4])).all()


def test_box_area_3d(boxes0_3d, boxes1_3d):
    areas0 = box_area(boxes0_3d)
    areas1 = box_area(boxes1_3d)

    assert (areas0 == torch.tensor([8, 8])).all()
    assert (areas1 == torch.tensor([8, 8, 8])).all()


def test_box_iou_2d(boxes0_2d, boxes1_2d):
    ious = box_iou(boxes0_2d, boxes1_2d)
    assert all([a == b for a, b in zip(ious.shape, (2, 3))])
    expected = torch.empty_like(ious).fill_((1.0 / 7.0))
    assert ious.allclose(expected)


def test_box_iou_3d(boxes0_3d, boxes1_3d):
    ious = box_iou(boxes0_3d, boxes1_3d)
    assert all([a == b for a, b in zip(ious.shape, (2, 3))])
    expected = torch.empty_like(ious).fill_((1.0 / 15.0))
    assert ious.allclose(expected)


def test_generalized_box_iou_2d(boxes0_2d, boxes1_2d):
    ious = generalized_box_iou(boxes0_2d, boxes1_2d)
    assert all([a == b for a, b in zip(ious.shape, (2, 3))])
    expected = torch.empty_like(ious).fill_((1.0 / 7.0) - (2.0 / 9.0))
    assert ious.allclose(expected)


def test_generalized_box_iou_3d(boxes0_3d, boxes1_3d):
    ious = generalized_box_iou(boxes0_3d, boxes1_3d)
    assert all([a == b for a, b in zip(ious.shape, (2, 3))])
    expected = torch.empty_like(ious).fill_((1.0 / 15.0) - (12.0 / 27.0))
    assert ious.allclose(expected)


def test_generalized_box_iou_3d_paired(boxes0_3d, boxes1_3d):
    ious = generalized_box_iou(boxes0_3d, boxes1_3d[1:])
    assert all([a == b for a, b in zip(ious.shape, (2, 2))])
    expected = torch.empty_like(ious).fill_((1.0 / 15.0) - (12.0 / 27.0))
    assert ious.allclose(expected)


def test_distance_box_iou_3d_paired(boxes0_3d, boxes1_3d):
    ious = distance_box_iou_3d_paired(boxes0_3d, boxes1_3d[1:])
    assert all([a == b for a, b in zip(ious.shape, (2, 2))])

    # iou = 1 / 15
    # cd^2 = 3
    # diag^2 = 27

    expected = torch.empty_like(ious).fill_(1 - 1 / 15 + 3 / 27)
    assert ious.allclose(expected)


@pytest.fixture(params=["a", "b"])
def arg(request):
    return request.getfuncargvalue(request.param)


@pytest.mark.parametrize(
    "boxes,similarity_fn",
    [
        ("boxes0_2d", box_iou),
        ("boxes1_2d", box_iou),
        ("boxes0_3d", box_iou),
        ("boxes1_3d", box_iou),
        ("boxes0_2d", generalized_box_iou),
        ("boxes1_2d", generalized_box_iou),
        ("boxes0_3d", generalized_box_iou),
        ("boxes1_3d", generalized_box_iou),
    ],
)
def test_sanity_check_iou_fn(boxes, similarity_fn, request):
    boxes = request.getfixturevalue(boxes)

    ious = similarity_fn(boxes, boxes)
    assert (ious == torch.ones_like(ious)).all()
    assert all([a == b for a, b in zip(ious.shape, (boxes.shape[0], boxes.shape[0]))])


def test_permute_boxes_2d():
    boxes = torch.rand((10, 4))
    new_boxes = permute_boxes(boxes)
    expected_boxes = boxes[:, [1, 0, 3, 2]]
    assert expected_boxes.allclose(new_boxes)


def test_permute_boxes_3d():
    boxes = torch.rand((10, 6))
    new_boxes = permute_boxes(boxes, (2, 0, 1))
    expected_boxes = boxes[:, [4, 0, 5, 2, 1, 3]]
    assert expected_boxes.allclose(new_boxes)


def test_cat_and_index(boxes0_3d, boxes1_3d):
    boxes, idx = cat_and_index(
        [boxes0_3d, torch.tensor([[]]).reshape(-1, 6), boxes1_3d]
    )
    torch.allclose(boxes, torch.cat([boxes0_3d, boxes1_3d], dim=0))
    torch.allclose(idx, torch.tensor([0, 0, 2, 2, 2], dtype=boxes0_3d.dtype))
