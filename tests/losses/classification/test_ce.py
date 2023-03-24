import pytest
import torch

from nndet.losses.classification.ce import BCELoss, CELoss
from nndet.losses.ops import one_hot_smooth_last


@pytest.fixture
def softmax_example():
    torch.manual_seed(0)
    preds = torch.rand(4, 24, 3)
    targets = torch.randint(0, 3, (4, 24))
    weight = torch.tensor([0.1, 1.0, 1.0])
    return preds, targets, weight


@pytest.fixture
def sigmoid_example():
    torch.manual_seed(0)
    preds = torch.rand(4, 24, 2)
    targets = torch.randint(0, 3, (4, 24))
    weight = None
    return preds, targets, weight


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_against_torch_ce_loss(softmax_example, reduction: str):
    preds, targets, weight = softmax_example
    nndet_loss = CELoss(weight=weight, loss_weight=2.0, reduction=reduction)
    torch_loss = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    nndet_loss_value = nndet_loss(preds, targets)
    torch_loss_value = 2 * torch_loss(preds.movedim(-1, 1), targets)

    assert torch.allclose(nndet_loss_value, torch_loss_value)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_against_torch_bce_loss(sigmoid_example, reduction: str):
    preds, targets, weight = sigmoid_example
    targets_one_hot = one_hot_smooth_last(targets, num_classes=3)[..., 1:]
    nndet_loss = BCELoss(pos_weight=weight, loss_weight=2.0, reduction=reduction)
    torch_loss = torch.nn.BCEWithLogitsLoss(pos_weight=weight, reduction=reduction)

    nndet_loss_value = nndet_loss(preds, targets)
    torch_loss_value = 2 * torch_loss(preds, targets_one_hot)

    assert torch.allclose(nndet_loss_value, torch_loss_value)
