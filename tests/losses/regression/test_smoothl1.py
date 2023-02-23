import math

import pytest
import torch

# import torch function to ensure same results as old nnDet implementation
from torch.nn.functional import smooth_l1_loss


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
