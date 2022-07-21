import math

import pytest
import torch

from nndet.losses.regression import GIoULoss
from nndet.losses.regression.diou import DIoULoss
from nndet.losses.regression.functional.smoothl1 import smooth_l1_loss
from nndet.losses.regression.giou import GIoULossPaired


@pytest.fixture
def inp():
    torch.manual_seed(0)
    return torch.rand(400, 1000)


@pytest.fixture
def target():
    torch.manual_seed(42)
    return torch.rand(400, 1000)


def test_functional_normal_beta(inp, target):
    inp = torch.tensor([0.2, 1.5])
    target = torch.tensor([0.3, 2.5])
    computed_loss = smooth_l1_loss(inp, target, beta=0.75, reduction="none")
    expected_loss = torch.tensor([(0.5 * 0.1**2 / 0.75), (1.0 - 0.5 * 0.75)])
    assert math.isclose((computed_loss - expected_loss).sum().item(), 0, abs_tol=1e-8)


def test_functional_l1_beta(inp, target):
    computed_loss = torch.nn.functional.l1_loss(inp, target, reduction="mean")
    expected_loss = smooth_l1_loss(inp, target, beta=1e-6, reduction="mean")
    assert math.isclose((computed_loss - expected_loss).item(), 0, abs_tol=1e-5)


def test_giou_loss():
    boxes0_3d = torch.tensor([[0, 0, 2, 2, 0, 1], [0, 0, 2, 2, 0, 1]]).float()
    boxes1_3d = torch.tensor([[1, 1, 3, 3, 0, 1], [1, 1, 3, 3, 0, 1]]).float()
    loss_fn = GIoULoss(reduction="sum", loss_weight=3.0)

    computed_loss = loss_fn(boxes0_3d, boxes1_3d)
    expected_loss = torch.tensor(-((1.0 / 7.0) - (2.0 / 9.0)) * 2 * 3)
    assert computed_loss.allclose(expected_loss)


def test_giou_paired_loss():
    boxes0_3d = torch.tensor([[0, 0, 2, 2, 0, 1], [0, 0, 2, 2, 0, 1]]).float()
    boxes1_3d = torch.tensor([[1, 1, 3, 3, 0, 1], [1, 1, 3, 3, 0, 1]]).float()
    loss_fn = GIoULossPaired(reduction="sum", loss_weight=3.0)

    computed_loss = loss_fn(boxes0_3d, boxes1_3d)
    expected_loss = torch.tensor(-((1.0 / 7.0) - (2.0 / 9.0)) * 2 * 3)
    assert computed_loss.allclose(expected_loss)


def test_diou_loss():
    boxes0_3d = torch.tensor([[0, 0, 2, 2, 0, 2], [0, 0, 2, 2, 0, 2]]).float()
    boxes1_3d = torch.tensor([[1, 1, 3, 3, 1, 3], [1, 1, 3, 3, 1, 3]]).float()
    loss_fn = DIoULoss(reduction="sum", loss_weight=3.0, eps=0)

    # iou = 1 / 15
    # cd^2 = 3
    # diag^2 = 27

    computed_loss = loss_fn(boxes0_3d, boxes1_3d)
    expected_loss = torch.tensor((1 - 1 / 15 + 3 / 27) * 2 * 3)
    print(computed_loss)
    print(expected_loss)
    assert computed_loss.allclose(expected_loss)
