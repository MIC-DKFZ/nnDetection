import pytest
import torch

from nndet.losses.mask.ce import BCEMaskLoss
from nndet.losses.ops import one_hot_smooth_last


@pytest.fixture
def sigmoid_example():
    torch.manual_seed(0)
    preds = torch.rand(4, 16, 16, 16, 3)
    targets = torch.randint(0, 3, (4, 16, 16, 16))
    targets_one_hot = one_hot_smooth_last(targets, num_classes=4)[..., 1:]  # .permute(0, -1, 1, 2, 3)
    weight = None
    return preds, targets_one_hot, weight


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_against_torch_bce_loss(sigmoid_example, reduction: str):
    preds, targets, weight = sigmoid_example

    nndet_loss = BCEMaskLoss(pos_weight=weight, loss_weight=2.0, reduction=reduction)
    torch_loss = torch.nn.BCEWithLogitsLoss(pos_weight=weight, reduction=reduction)

    nndet_loss_value = nndet_loss(preds, targets)
    torch_loss_value = 2 * torch_loss(preds, targets)

    assert torch.allclose(nndet_loss_value, torch_loss_value)
