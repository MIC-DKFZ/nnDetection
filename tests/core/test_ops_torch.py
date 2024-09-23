import math

import pytest
import torch

import nndet.core.ops_torch as ops_torch
from nndet.core.boxes import *


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
    return torch.tensor([[1, 1, 3, 3, 1, 3], [1, 1, 3, 3, 1, 3], [1, 1, 3, 3, 1, 3]]).float()


def check_ious(boxes: torch.Tensor, similarity_fn):
    """
    Computes iou from boxes to boxes -> should always result in 1
    """


def test_box_area_2d(boxes0_2d, boxes1_2d):
    areas0 = ops_torch.box_area(boxes0_2d)
    areas1 = ops_torch.box_area(boxes1_2d)

    assert (areas0 == torch.tensor([4, 4])).all()
    assert (areas1 == torch.tensor([4, 4, 4])).all()


def test_box_area_3d(boxes0_3d, boxes1_3d):
    areas0 = ops_torch.box_area(boxes0_3d)
    areas1 = ops_torch.box_area(boxes1_3d)

    assert (areas0 == torch.tensor([8, 8])).all()
    assert (areas1 == torch.tensor([8, 8, 8])).all()


def test_box_inter_2d(boxes0_2d, boxes1_2d):
    intersections = ops_torch.box_inter(boxes0_2d, boxes1_2d)
    assert all([a == b for a, b in zip(intersections.shape, (2, 3))])
    expected = torch.empty_like(intersections).fill_(1.0)
    assert intersections.allclose(expected)


def test_box_inter_3d(boxes0_3d, boxes1_3d):
    intersections = ops_torch.box_inter(boxes0_3d, boxes1_3d)
    assert all([a == b for a, b in zip(intersections.shape, (2, 3))])
    expected = torch.empty_like(intersections).fill_(1.0)
    assert intersections.allclose(expected)


def test_box_iou_2d(boxes0_2d, boxes1_2d):
    ious = ops_torch.box_iou(boxes0_2d, boxes1_2d)
    assert all([a == b for a, b in zip(ious.shape, (2, 3))])
    expected = torch.empty_like(ious).fill_((1.0 / 7.0))
    assert ious.allclose(expected)


def test_box_iou_3d(boxes0_3d, boxes1_3d):
    ious = ops_torch.box_iou(boxes0_3d, boxes1_3d)
    assert all([a == b for a, b in zip(ious.shape, (2, 3))])
    expected = torch.empty_like(ious).fill_((1.0 / 15.0))
    assert ious.allclose(expected)


def test_generalized_box_iou_2d(boxes0_2d, boxes1_2d):
    ious = ops_torch.generalized_box_iou(boxes0_2d, boxes1_2d)
    assert all([a == b for a, b in zip(ious.shape, (2, 3))])
    expected = torch.empty_like(ious).fill_((1.0 / 7.0) - (2.0 / 9.0))
    assert ious.allclose(expected)


def test_generalized_box_iou_3d(boxes0_3d, boxes1_3d):
    ious = ops_torch.generalized_box_iou(boxes0_3d, boxes1_3d)
    assert all([a == b for a, b in zip(ious.shape, (2, 3))])
    expected = torch.empty_like(ious).fill_((1.0 / 15.0) - (12.0 / 27.0))
    assert ious.allclose(expected)


def test_generalized_box_iou_3d_paired(boxes0_3d, boxes1_3d):
    ious = ops_torch.generalized_box_iou_paired(boxes0_3d, boxes1_3d[1:])
    assert all([a == b for a, b in zip(ious.shape, (2, 2))])
    expected = torch.empty_like(ious).fill_((1.0 / 15.0) - (12.0 / 27.0))
    assert ious.allclose(expected)


def test_distance_box_iou_3d_paired(boxes0_3d, boxes1_3d):
    ious = ops_torch.distance_box_iou_paired(boxes0_3d, boxes1_3d[1:])
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
        ("boxes0_2d", ops_torch.box_iou),
        ("boxes1_2d", ops_torch.box_iou),
        ("boxes0_3d", ops_torch.box_iou),
        ("boxes1_3d", ops_torch.box_iou),
        ("boxes0_2d", ops_torch.generalized_box_iou),
        ("boxes1_2d", ops_torch.generalized_box_iou),
        ("boxes0_3d", ops_torch.generalized_box_iou),
        ("boxes1_3d", ops_torch.generalized_box_iou),
    ],
)
def test_sanity_check_iou_fn(boxes, similarity_fn, request):
    boxes = request.getfixturevalue(boxes)

    ious = similarity_fn(boxes, boxes)
    assert (ious == torch.ones_like(ious)).all()
    assert all([a == b for a, b in zip(ious.shape, (boxes.shape[0], boxes.shape[0]))])


def test_permute_boxes_2d():
    boxes = torch.rand((10, 4))
    new_boxes = ops_torch.permute_boxes(boxes)
    expected_boxes = boxes[:, [1, 0, 3, 2]]
    assert expected_boxes.allclose(new_boxes)


def test_permute_boxes_3d():
    boxes = torch.rand((10, 6))
    new_boxes = ops_torch.permute_boxes(boxes, (2, 0, 1))
    expected_boxes = boxes[:, [4, 0, 5, 2, 1, 3]]
    assert expected_boxes.allclose(new_boxes)


def test_cat_and_index(boxes0_3d, boxes1_3d):
    boxes, idx = ops_torch.cat_and_index([boxes0_3d, torch.tensor([[]]).reshape(-1, 6), boxes1_3d])
    assert torch.allclose(boxes, torch.cat([boxes0_3d, boxes1_3d], dim=0))
    assert torch.allclose(idx, torch.tensor([0, 0, 2, 2, 2], dtype=boxes0_3d.dtype))


def test_clip_boxes_2d_inplace():
    boxes = torch.tensor([[-1, -2, 102, 200]])
    boxes = ops_torch.clip_boxes_to_image_(boxes, (100, 200))
    expected = torch.tensor([[-1, -1, 100, 200]])
    assert (expected == boxes).all()


def test_clip_boxes_3d_inplace():
    boxes = torch.tensor([[-1, -2, 102, 200, -5, 30]])
    boxes = ops_torch.clip_boxes_to_image_(boxes, (100, 200, 16))
    expected = torch.tensor([[-1, -1, 100, 200, -1, 16]])
    assert (expected == boxes).all()


def test_clip_boxes_2d():
    boxes = torch.tensor([[-1, -2, 102, 200]])
    boxes = ops_torch.clip_boxes_to_image(boxes, (100, 200))
    expected = torch.tensor([[-1, -1, 100, 200]])
    assert (expected == boxes).all()


def test_clip_boxes_3d():
    boxes = torch.tensor([[-1, -2, 102, 200, -5, 30]])
    boxes = ops_torch.clip_boxes_to_image(boxes, (100, 200, 16))
    expected = torch.tensor([[-1, -1, 100, 200, -1, 16]])
    assert (expected == boxes).all()


def test_box_ccddcd2cccddd_2d(boxes0_2d):
    expected_output = torch.clone(boxes0_2d)
    output = ops_torch.box_ccddcd2cccddd(boxes0_2d)
    assert torch.allclose(expected_output, output)


def test_box_ccddcd2cccddd_3d(boxes0_3d):
    expected_output = torch.tensor([[0, 0, 0, 2, 2, 2], [0, 0, 0, 2, 2, 2]]).float()
    output = ops_torch.box_ccddcd2cccddd(boxes0_3d)
    assert torch.allclose(expected_output, output)


def test_box_cccddd2ccddcd_2d(boxes0_2d):
    expected_output = torch.clone(boxes0_2d)
    output = ops_torch.box_ccddcd2cccddd(boxes0_2d)
    assert torch.allclose(expected_output, output)


def test_box_cccddd2ccddcd_3d(boxes0_3d):
    boxes_input = torch.tensor([[0, 0, 0, 2, 2, 2], [0, 0, 0, 2, 2, 2]]).float()
    output = ops_torch.box_cccddd2ccddcd(boxes_input)
    assert torch.allclose(boxes0_3d, output)


def test_box_center2point_format_2d(boxes0_2d):
    boxes_input = torch.tensor([[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]]).float()
    output = ops_torch.box_center2point_format(boxes_input)
    assert torch.allclose(boxes0_2d, output)


def test_box_point2center_format_2d(boxes0_2d):
    expected_output = torch.tensor([[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]]).float()
    output = ops_torch.box_point2center_format(boxes0_2d)
    assert torch.allclose(expected_output, output)


def test_box_center2point_format_3d(boxes0_3d):
    boxes_input = torch.tensor(
        [
            [1.0, 1.0, 2.0, 2.0, 1.0, 2.0],
            [1.0, 1.0, 2.0, 2.0, 1.0, 2.0],
        ]
    ).float()
    output = ops_torch.box_center2point_format(boxes_input)
    assert torch.allclose(boxes0_3d, output)


def test_box_point2center_format_3d(boxes0_3d):
    expected_output = torch.tensor(
        [
            [1.0, 1.0, 2.0, 2.0, 1.0, 2.0],
            [1.0, 1.0, 2.0, 2.0, 1.0, 2.0],
        ]
    ).float()
    output = ops_torch.box_point2center_format(boxes0_3d)
    assert torch.allclose(expected_output, output)


def test_box_point_rescale_with_size_2d(boxes0_2d):
    scaled_boxes = ops_torch.box_point_rescale_with_size(boxes0_2d, (1, 2))
    expected_output = torch.tensor([[0, 0, 2, 4], [0, 0, 2, 4]]).float()
    assert torch.allclose(expected_output, scaled_boxes)


def test_box_point_rescale_with_size_3d(boxes0_3d):
    scaled_boxes = ops_torch.box_point_rescale_with_size(boxes0_3d, (1, 2, 3))
    expected_output = torch.tensor([[0, 0, 2, 4, 0, 6], [0, 0, 2, 4, 0, 6]]).float()
    assert torch.allclose(expected_output, scaled_boxes)


def test_box_point_rescale_with_size_3d_extra_batched(boxes0_3d):
    boxes0_3d_extra_batched = torch.stack([boxes0_3d, boxes0_3d])
    scaled_boxes = ops_torch.box_point_rescale_with_size(boxes0_3d_extra_batched, (1, 2, 3), extra_batched=True)
    expected_output = torch.tensor([[0, 0, 2, 4, 0, 6], [0, 0, 2, 4, 0, 6]]).float()
    expected_output_extra_batched = torch.stack([expected_output, expected_output])
    assert torch.allclose(expected_output_extra_batched, scaled_boxes)


def test_box_point_norm_with_size_2d(boxes0_2d):
    scaled_boxes = ops_torch.box_point_norm_with_size(boxes0_2d, (1, 2))
    expected_output = torch.tensor([[0, 0, 2, 1], [0, 0, 2, 1]]).float()
    assert torch.allclose(expected_output, scaled_boxes)


def test_box_point_norm_with_size_3d(boxes0_3d):
    scaled_boxes = ops_torch.box_point_norm_with_size(boxes0_3d, (1, 2, 0.5))
    expected_output = torch.tensor([[0, 0, 2, 1, 0, 4], [0, 0, 2, 1, 0, 4]]).float()
    assert torch.allclose(expected_output, scaled_boxes)


def test_inverse_sigmoid():
    torch.random.manual_seed(42)
    x = torch.rand(100) + 0.0001
    x_sigmoid = torch.nn.functional.sigmoid(x)
    x_inv = ops_torch.inverse_sigmoid(x_sigmoid, eps=0)
    assert torch.allclose(x, x_inv, atol=1e-5)


def test_inverse_sigmoid_module():
    torch.random.manual_seed(42)
    x = torch.rand(100) + 0.0001
    x_sigmoid = torch.nn.functional.sigmoid(x)
    module = ops_torch.InverseSigmoid(eps=0)
    x_inv = module(x_sigmoid)
    assert torch.allclose(x, x_inv, atol=1e-5)
