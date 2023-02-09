import pytest
import torch

# classification
from nndet.losses.classification.ce import BCELoss, CELoss
from nndet.losses.classification.focal import AsymmetricBFocalLoss, BFocalLoss
from nndet.losses.classification.poly1 import Poly1BCEWithLogits, Poly1BFocalLoss

# mask
from nndet.losses.mask.ce import BCEMaskLoss
from nndet.losses.regression.diou import DIoULoss
from nndet.losses.regression.giou import GIoULoss, GIoULossPaired

# regression
from nndet.losses.regression.smoothl1 import SmoothL1Loss

# segmentation
from nndet.losses.segmentation.ce import BCESegLoss, CESegLoss
from nndet.losses.segmentation.dice import SoftDiceSegLoss
from nndet.losses.segmentation.topk import TopKBCESegLoss, TopKCESegLoss

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
CLS_TARGET = torch.ones(10, dtype=torch.float)
CLS_TARGET_SANITY = torch.zeros(10, dtype=torch.float)

SEG_PRED = torch.zeros(1, 3, 10, dtype=torch.float)
SEG_TARGET = torch.ones(1, 10, dtype=torch.float)
SEG_TARGET_SANITY = torch.zeros(1, 10, dtype=torch.float)

MASK_PRED = torch.zeros(1, 3, 10, dtype=torch.float)
MASK_TARGET = torch.ones(1, 3, 10, dtype=torch.float)
MASK_TARGET_SANITY = torch.zeros(1, 3, 10, dtype=torch.float)


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
TEST_CLASSIFICATION_LOSSES = [
    CELoss(reduction="mean"),
    BCELoss(reduction="mean"),
    BFocalLoss(reduction="mean"),
    AsymmetricBFocalLoss(reduction="mean"),
    Poly1BCEWithLogits(reduction="mean"),
    Poly1BFocalLoss(reduction="mean"),
    # other reductions
    CELoss(reduction="mean_last_sum"),
    BCELoss(reduction="mean_last_sum"),
]
TEST_SEGMENTATION_LOSSES = [
    CESegLoss(reduction="mean"),
    BCESegLoss(reduction="mean"),
    SoftDiceSegLoss(reduction="mean"),
    TopKCESegLoss(topk=0.1),
    TopKBCESegLoss(topk=0.1),
]
TEST_MASK_LOSSES = [
    BCEMaskLoss(reduction="mean"),
]


TEST_CASES = (
    [(l, BOXES_PRED, BOXES_TARGET) for l in TEST_REGRESSION_LOSSES]
    + [(l, CLS_PRED, CLS_TARGET) for l in TEST_CLASSIFICATION_LOSSES]
    + [(l, SEG_PRED, SEG_TARGET) for l in TEST_SEGMENTATION_LOSSES]
    + [(l, MASK_PRED, MASK_TARGET) for l in TEST_MASK_LOSSES]
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
