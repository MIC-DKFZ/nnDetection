import pytest
import math

import torch
from nndet.core.boxes import BoxCoderND


@pytest.fixture
def coder2d():
    return BoxCoderND([1., 1., 1., 1.])
    

@pytest.fixture
def coder3d():
    return BoxCoderND([1., 1., 1., 1., 1., 1.])


def test_box_coder_2d_decode(coder2d):
    box_ref = [torch.tensor([[0, 0, 10, 10]])]
    box_delta = torch.tensor([1, 1, math.log(1), math.log(1)])
    box = coder2d.decode(box_delta, box_ref)
    expected = torch.tensor([[[10, 10, 20, 20]]])
    assert math.isclose((box - expected).sum(), 0)


def test_box_coder_2d_encode(coder2d):
    box_anchor = [torch.tensor([[0, 0, 10, 10]]).float()]
    box_gt = [torch.tensor([[10, 10, 20, 20]]).float()]
    delta = coder2d.encode(box_gt, box_anchor)[0]
    expected = torch.tensor([[1, 1, math.log(1), math.log(1)]])
    assert (math.isclose((delta - expected).sum(), 0))


def test_box_coder_2d_encode_decode(coder2d):
    box_anchor = [torch.tensor([[0, 0, 10, 10]]).float()]
    box_gt = [torch.tensor([[10, 10, 20, 20]]).float()]
    delta = coder2d.encode(box_gt, box_anchor)[0]
    box = coder2d.decode(delta, box_anchor)
    assert (math.isclose((box - box_gt[0]).sum(), 0))


def test_box_coder_3d_decode(coder3d):
    box_ref = [torch.tensor([[0, 0, 10, 10, 0, 10]])]
    box_delta = torch.tensor([1, 1, math.log(1), math.log(1), 1, math.log(1)])
    box = coder3d.decode(box_delta, box_ref)
    expected = torch.tensor([[[10, 10, 20, 20, 10, 20]]])
    assert (math.isclose((box - expected).sum(), 0))


def test_box_coder_3d_encode(coder3d):
    box_anchor = [torch.tensor([[0, 0, 10, 10, 0, 10]]).float()]
    box_gt = [torch.tensor([[10, 10, 20, 20, 10, 20]]).float()]
    delta = coder3d.encode(box_gt, box_anchor)[0]
    expected = torch.tensor([[1, 1, math.log(1), math.log(1), 1, math.log(1)]])
    assert (math.isclose((delta - expected).sum(), 0))


def test_box_coder_3d_encode_decode(coder3d):
    box_anchor = [torch.tensor([[0, 0, 10, 10, 0, 10]]).float()]
    box_gt = [torch.tensor([[10, 10, 20, 20, 10, 20]]).float()]
    delta = coder3d.encode(box_gt, box_anchor)[0]
    box = coder3d.decode(delta, box_anchor)
    assert (math.isclose((box - box_gt[0]).sum(), 0))
