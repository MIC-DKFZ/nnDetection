import math

import pytest
import torch

from nndet.core.boxes import BoxCoderND


@pytest.fixture
def coder2d():
    return BoxCoderND([1.0, 1.0, 1.0, 1.0])


@pytest.fixture
def coder3d():
    return BoxCoderND([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


CODE_TESTCASES = [
    (
        [[0, 0, 10, 10, 0, 10]],
        [[10, 10, 20, 20, 10, 20]],
        [[1, 1, math.log(1), math.log(1), 1, math.log(1)]],
    ),
    (
        [[10, 0, 20, 10, 0, 10]],
        [[10, 10, 20, 20, 10, 20]],
        [[0, 1, 0, math.log(1), 1, math.log(1)]],
    ),
    (
        [[0, 10, 10, 20, 0, 10]],
        [[10, 10, 20, 20, 10, 20]],
        [[1, 0, math.log(1), 0, 1, math.log(1)]],
    ),
    (
        [[0, 0, 10, 10, 10, 20]],
        [[10, 10, 20, 20, 10, 20]],
        [[1, 1, math.log(1), math.log(1), 0, 0]],
    ),
]


ENCODE_DECODE_TESTCASES = [
    ([[0, 0, 10, 10, 0, 10]], [[10, 10, 20, 20, 10, 20]]),
    ([[0, 0, 20, 10, 0, 10]], [[10, 10, 20, 20, 10, 20]]),
    ([[0, 0, 10, 20, 0, 10]], [[10, 10, 20, 20, 10, 20]]),
    ([[0, 0, 10, 10, 0, 20]], [[10, 10, 20, 20, 10, 20]]),
    ([[5, 0, 10, 10, 0, 20]], [[10, 10, 20, 20, 10, 20]]),
    ([[0, 5, 10, 10, 0, 20]], [[10, 10, 20, 20, 10, 20]]),
    ([[0, 0, 10, 10, 5, 20]], [[10, 10, 20, 20, 10, 20]]),
    ([[5, 0, 15, 10, 0, 20]], [[10, 10, 20, 20, 10, 20]]),
    ([[0, 5, 15, 10, 0, 20]], [[10, 10, 20, 20, 10, 20]]),
    ([[0, 5, 10, 10, 0, 15]], [[10, 10, 20, 20, 10, 20]]),
    ([[0, 0, 10, 10, 0, 10]], [[0, 0, 20, 20, 0, 20]]),
    ([[5, 0, 10, 10, 0, 10]], [[0, 0, 20, 20, 0, 20]]),
    ([[0, 5, 10, 10, 0, 10]], [[0, 0, 20, 20, 0, 20]]),
    ([[0, 0, 10, 10, 5, 10]], [[0, 0, 20, 20, 0, 20]]),
]


def test_box_coder_2d_decode(coder2d):
    box_ref = [torch.tensor([[0, 0, 10, 10]])]
    box_delta = torch.tensor([1, 1, math.log(1), math.log(1)])
    box = coder2d.decode(box_delta, box_ref)
    expected = torch.tensor([[[10, 10, 20, 20]]])
    assert math.isclose((box - expected).sum(), 0)


def test_box_coder_2d_decode_class_specific(coder2d):
    """
    Bounding box delta per class per anchor
    """
    box_ref = [torch.tensor([[0, 0, 10, 10]])]
    box_delta = torch.tensor([1, 1, math.log(1), math.log(1), 2, 2, math.log(3), math.log(3)])
    box = coder2d.decode(box_delta, box_ref)
    expected = torch.tensor([[[10, 10, 20, 20, 10, 10, 40, 40]]])
    assert math.isclose((box - expected).sum(), 0)


def test_box_coder_2d_encode(coder2d):
    box_anchor = [torch.tensor([[0, 0, 10, 10]]).float()]
    box_gt = [torch.tensor([[10, 10, 20, 20]]).float()]
    delta = coder2d.encode(box_gt, box_anchor)[0]
    expected = torch.tensor([[1, 1, math.log(1), math.log(1)]])
    assert math.isclose((delta - expected).sum(), 0)


def test_box_coder_2d_encode_decode(coder2d):
    box_anchor = [torch.tensor([[0, 0, 10, 10]]).float()]
    box_gt = [torch.tensor([[10, 10, 20, 20]]).float()]
    delta = coder2d.encode(box_gt, box_anchor)[0]
    box = coder2d.decode(delta, box_anchor)
    assert math.isclose((box - box_gt[0]).sum(), 0)


@pytest.mark.parametrize("box_anchor_list,box_gt_list,box_expected_list", CODE_TESTCASES)
def test_box_coder_3d_encode(coder3d, box_anchor_list, box_gt_list, box_expected_list):
    box_anchor = [torch.tensor(box_anchor_list).float()]
    box_gt = [torch.tensor(box_gt_list).float()]
    delta = coder3d.encode(box_gt, box_anchor)[0]
    expected = torch.tensor(box_expected_list)

    assert tuple(delta.shape) == tuple(expected.shape)
    assert torch.isclose(delta, expected).all()


@pytest.mark.parametrize("box_ref_list,box_expected_list,box_delta_list", CODE_TESTCASES)
def test_box_coder_3d_decode(coder3d, box_ref_list, box_expected_list, box_delta_list):
    box_ref = [torch.tensor(box_ref_list).float()]
    box_delta = torch.tensor(box_delta_list).float()
    box = coder3d.decode(box_delta, box_ref)
    expected = torch.tensor(box_expected_list).float()
    assert tuple(expected.shape) == tuple(box.shape)
    assert torch.isclose(box, expected).all()


@pytest.mark.parametrize("box_anchor_list,box_gt_list", ENCODE_DECODE_TESTCASES)
def test_box_coder_3d_encode_decode(coder3d, box_anchor_list, box_gt_list):
    box_anchor = [torch.tensor(box_anchor_list).float()]
    box_gt = [torch.tensor(box_gt_list).float()]
    delta = coder3d.encode(box_gt, box_anchor)[0]
    box = coder3d.decode(delta, box_anchor)

    assert tuple(box.shape) == tuple(box_gt[0].shape)
    assert torch.isclose(box, box_gt[0]).all()


def test_box_coder_3d_decode_class_specific(coder3d):
    """
    Bounding box delta per class per anchor
    """
    box_ref = [torch.tensor([[0, 0, 10, 10, 0, 10]])]
    box_delta = torch.tensor(
        [
            1,
            1,
            math.log(1),
            math.log(1),
            1,
            math.log(1),
            2,
            2,
            math.log(3),
            math.log(3),
            2,
            math.log(3),
        ]
    )
    box = coder3d.decode(box_delta, box_ref)
    expected = torch.tensor([[[10, 10, 20, 20, 10, 20, 10, 10, 40, 40, 10, 40]]])
    assert math.isclose((box - expected).sum(), 0)
