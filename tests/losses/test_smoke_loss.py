import pytest
import torch

from nndet.losses.classification import (
    AsymmetricFocalLossWithLogits,
    BCEWithLogitsLoss,
    BCEWithLogitsLossOneHot,
    CrossEntropyLoss,
    FocalLossWithLogits,
)
from nndet.losses.classification.bce import BinaryCrossEntropyLoss
from nndet.losses.classification.poly1 import (
    Poly1BCEWithLogits,
    Poly1FocalLossWithLogits,
)
from nndet.losses.regression import GIoULoss, SmoothL1Loss
from nndet.losses.regression.diou import DIoULoss
from nndet.losses.regression.giou import GIoULossPaired
from nndet.losses.segmentation import SoftDiceLoss, TopKLoss, TopKLossSigmoid

"""
Add all nnDetection Losses to this list
Prediction and Targets should be float

Smoke-Tests cover:
    - simple loss_weight test
    - correct casting of half tensors to float tensors if loss_fp32=True
    - basic backward() scenario
"""

BOXES_PRED = torch.Tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 1.0]])
BOXES_TARGET = torch.Tensor([[1.0, 1.0, 2.0, 3.0, 1.0, 4.0]])
BOXES_TARGET_SANITY = torch.Tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 1.0]])

CLS_PRED = torch.zeros(10, 3, dtype=torch.float)
CLS_SIGMOID_TARGET = torch.ones(10, 3, dtype=torch.float)
CLS_SIGMOID_TARGET_SANITY = torch.zeros(10, 3, dtype=torch.float)
CLS_LABEL_TARGET = torch.ones(10, dtype=torch.float)
CLS_LABEL_TARGET_SANITY = torch.zeros(10, dtype=torch.float)

SEG_PRED = torch.zeros(1, 3, 10, dtype=torch.float)
SEG_SIGMOID_TARGET = torch.ones(1, 3, 10, dtype=torch.float)
SEG_SIGMOID_TARGET_SANITY = torch.zeros(1, 3, 10, dtype=torch.float)
SEG_LABEL_TARGET = torch.ones(1, 10, dtype=torch.float)
SEG_LABEL_TARGET_SANITY = torch.zeros(1, 10, dtype=torch.float)


TEST_REGRESSION_LOSSES = [
    SmoothL1Loss(beta=1.0, reduction="mean"),
    GIoULoss(reduction="mean"),
    GIoULossPaired(reduction="mean"),
    DIoULoss(reduction="mean"),
]
TEST_REGRESSION_LOSSES_WITH_SANITY = [
    SmoothL1Loss(beta=1.0, reduction="mean"),
    DIoULoss(reduction="mean", eps=1e-12),
]
TEST_CLASSIFICATION_LABEL_LOSSES = [
    FocalLossWithLogits(reduction="mean"),
    AsymmetricFocalLossWithLogits(reduction="mean"),
    CrossEntropyLoss(reduction="mean"),
    BCEWithLogitsLossOneHot(reduction="mean"),
    Poly1BCEWithLogits(reduction="mean"),
    Poly1FocalLossWithLogits(reduction="mean"),
    BinaryCrossEntropyLoss(reduction="mean"),
    # test losses with custom reduction (only sigmoid based losses)
    FocalLossWithLogits(reduction="mean_d1_sum"),
    AsymmetricFocalLossWithLogits(reduction="mean_d1_sum"),
    Poly1BCEWithLogits(reduction="mean_d1_sum"),
    Poly1FocalLossWithLogits(reduction="mean_d1_sum"),
    BinaryCrossEntropyLoss(reduction="mean_d1_sum"),
]
TEST_CLASSIFICATION_SIGMOID_LOSSES = [
    BCEWithLogitsLoss(reduction="mean"),
]
TEST_SEGMENTATION_LABEL_LOSSES = [
    TopKLossSigmoid(num_classes=3, topk=0.0),
    SoftDiceLoss(reduction="mean"),
    TopKLoss(topk=0.1),
]
TEST_SEGMENTATION_SIGMOID_LOSSES = []


TEST_CASES = (
    [(l, BOXES_PRED, BOXES_TARGET) for l in TEST_REGRESSION_LOSSES]
    + [(l, CLS_PRED, CLS_LABEL_TARGET) for l in TEST_CLASSIFICATION_LABEL_LOSSES]
    + [(l, CLS_PRED, CLS_SIGMOID_TARGET) for l in TEST_CLASSIFICATION_SIGMOID_LOSSES]
    + [(l, SEG_PRED, SEG_LABEL_TARGET) for l in TEST_SEGMENTATION_LABEL_LOSSES]
    + [(l, SEG_PRED, SEG_SIGMOID_TARGET) for l in TEST_SEGMENTATION_SIGMOID_LOSSES]
)
TEST_CASES_SANITY = [(l, BOXES_PRED, BOXES_TARGET_SANITY) for l in TEST_REGRESSION_LOSSES_WITH_SANITY]


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


@pytest.mark.parametrize("loss_fn,pred,target", TEST_CASES_SANITY)
def test_sanity(loss_fn, pred, target):
    pred.requires_grad = True
    loss = loss_fn(pred, target)
    assert torch.isclose(loss, torch.tensor([0], dtype=loss.dtype))
