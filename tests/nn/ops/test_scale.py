import pytest
import torch
from numpy import require

from nndet.nn.ops.scale import Scale, ScalePerDim


@pytest.fixture
def reg_pred():
    reg = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]]
    )
    return reg


@pytest.fixture
def reg_target():
    reg = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]]
    )
    return reg


def test_scale(reg_pred, reg_target):
    module = Scale(scale=2.0)
    reg_final = module(reg_pred)

    loss = (reg_target - reg_final).sum()

    assert torch.allclose(loss, torch.tensor([-42.0]))

    loss.backward()
    assert torch.allclose(module.scale.grad, torch.tensor([-42.0]))


def test_scale_per_dim_2d():
    reg_pred = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]])
    reg_target = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]])

    module = ScalePerDim(scale=[2.0, 4.0])
    reg_final = module(reg_pred)

    loss = (reg_target - reg_final).sum()
    assert torch.allclose(loss, torch.tensor([-44.0]))  # (1 + 3 + 6 + 12) * 2

    loss.backward()
    assert torch.allclose(module.scale.grad, torch.tensor([-8.0, -12.0]))


def test_scale_per_dim_3d(reg_pred, reg_target):
    module = ScalePerDim(scale=[2.0, 4.0, 8.0])
    reg_final = module(reg_pred)

    loss = (reg_target - reg_final).sum()
    assert torch.allclose(
        loss, torch.tensor([-198.0])
    )  # (1 + 3 + 6 + 12 + 35 + 42) * 2

    loss.backward()
    assert torch.allclose(module.scale.grad, torch.tensor([-8.0, -12.0, -22.0]))
