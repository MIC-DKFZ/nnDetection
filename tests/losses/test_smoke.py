import pytest
import torch

from nndet.losses.classification import (
    AsymmetricFocalLossWithLogits,
    BCEWithLogitsLoss,
    BCEWithLogitsLossOneHot,
    CrossEntropyLoss,
    FocalLossWithLogits,
)
from nndet.losses.regression import GIoULoss, SmoothL1Loss
from nndet.losses.segmentation import SoftDiceLoss, TopKLoss, TopKLossSigmoid

"""
Add all nnDetection Losses to this list
Prediction and Targets should be float

Smoke-Tests cover:
    - simple loss_weight test
    - correct casting of half tensors to float tensors if loss_fp32=True
    - basic backward() scenario
"""

BOXES_PRED = torch.Tensor([[0.0, 0.0, 1.0, 1.0]])
BOXES_TARGET = torch.Tensor([[1.0, 1.0, 2.0, 3.0]])


TEST_CASES = [
    (
        FocalLossWithLogits(reduction="mean"),
        torch.zeros(1, 3, 10, dtype=torch.float),
        torch.ones(1, 10, dtype=torch.float),
    ),
    (
        AsymmetricFocalLossWithLogits(reduction="mean"),
        torch.zeros(1, 3, 10, dtype=torch.float),
        torch.ones(1, 10, dtype=torch.float),
    ),
    (
        CrossEntropyLoss(reduction="mean"),
        torch.zeros(1, 3, 10, dtype=torch.float),
        torch.ones(1, 10, dtype=torch.float),
    ),
    (
        BCEWithLogitsLoss(reduction="mean"),
        torch.zeros(1, 3, 10, dtype=torch.float),
        torch.ones(1, 3, 10, dtype=torch.float),
    ),
    (
        BCEWithLogitsLossOneHot(num_classes=3, reduction="mean"),
        torch.zeros(10, 3, dtype=torch.float),
        torch.ones(10, dtype=torch.float),
    ),
    (
        SmoothL1Loss(beta=1.0, reduction="mean"),
        torch.zeros(10, dtype=torch.float),
        torch.ones(10, dtype=torch.float),
    ),
    (
        GIoULoss(reduction="mean"),
        BOXES_PRED,
        BOXES_TARGET,
    ),
    (
        SoftDiceLoss(reduction="mean"),
        torch.zeros(1, 3, 10, dtype=torch.float),
        torch.ones(1, 10, dtype=torch.float),
    ),
    (
        TopKLoss(topk=0.1),
        torch.zeros(1, 3, 10, dtype=torch.float),
        torch.ones(1, 10, dtype=torch.float),
    ),
    (
        TopKLossSigmoid(num_classes=3, topk=0.1),
        torch.zeros(1, 3, 10, dtype=torch.float),
        torch.ones(1, 10, dtype=torch.float),
    ),
]


@pytest.mark.parametrize("loss_fn,pred,target", TEST_CASES)
def test_loss_weight(loss_fn, pred, target):
    with torch.no_grad():
        base_val = loss_fn(pred, target)
    loss_fn.loss_weight *= 2.0

    with torch.no_grad():
        scaled_val = loss_fn(pred, target)
    assert scaled_val.allclose(2 * base_val)


@pytest.mark.parametrize("loss_fn,pred,target", TEST_CASES)
def test_loss_fp32(loss_fn, pred, target):
    loss_fn.loss_fp32 = True

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            base_val = loss_fn(pred.half(), target.half())

    assert base_val.dtype == torch.float32


@pytest.mark.parametrize("loss_fn,pred,target", TEST_CASES)
def test_backward(loss_fn, pred, target):
    pred.requires_grad = True
    base_val = loss_fn(pred, target)
    base_val.backward()
