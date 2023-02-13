from typing import Type

import pytest
import torch

from nndet.core.boxes.coder import BoxCoderND
from nndet.nn.heads.classifier.roi import BCEConvRoIClassifier
from nndet.nn.heads.comb.base import RoIHead
from nndet.nn.heads.comb.roi import RoIBoxHead
from nndet.nn.heads.regressor.roi import (
    L1ConvRoIAgnosticRegressor,
    L1ConvRoISpecificRegressor,
)
from nndet.nn.layers.conv import ConvInstanceLReLU
from nndet.nn.layers.wrapper import Generator

NUM_CLASSES = 2
DIM = 3
N = 8
INPUT_SIZE_TENSOR = (N, 16, 4, 4, 4)
INPUT_SIZE_CONFIG = (4, 4, 4)
TARGET_SIZE_CLS = N
TARGET_SIZE_REG = (N, DIM * 2)

EXAMPLE_CONFIG = {
    "conv": Generator(ConvInstanceLReLU, 3),
    "in_channels": 16,
    "internal_channels": 32,
    "num_convs": 1,
    "input_size": INPUT_SIZE_CONFIG,
    "num_classes": NUM_CLASSES,
}
PROPOSALS = [torch.tensor([[1.0, 1.0, 2.0, 2.0, 1.0, 2.0]]).expand(N // 4, DIM * 2) for _ in range(4)]


@pytest.fixture
def classifier():
    return BCEConvRoIClassifier(
        **EXAMPLE_CONFIG,
        add_norm=True,
    )


@pytest.fixture
def regressor_agnostic():
    return L1ConvRoIAgnosticRegressor(
        **EXAMPLE_CONFIG,
        add_norm=True,
    )


@pytest.fixture
def regressor_specific():
    return L1ConvRoISpecificRegressor(
        **EXAMPLE_CONFIG,
        add_norm=True,
    )


@pytest.fixture
def coder():
    return BoxCoderND([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


@pytest.mark.parametrize("module_cls", [RoIBoxHead])
def test_head_agnostic(module_cls: Type[RoIHead], classifier, regressor_agnostic, coder):
    head: RoIHead = module_cls(
        classifier=classifier,
        regressor=regressor_agnostic,
        coder=coder,
    )
    roi_map = torch.zeros(INPUT_SIZE_TENSOR, dtype=torch.float)
    preds = head(roi_map)

    assert "box_deltas" in preds
    assert "box_logits" in preds
    assert tuple(preds["box_deltas"].shape) == (N, DIM * 2)
    assert tuple(preds["box_logits"].shape) == (N, NUM_CLASSES)

    preds_post = head.postprocess_for_inference(preds, PROPOSALS)
    assert "pred_boxes" in preds_post
    assert "pred_probs" in preds_post
    assert tuple(preds_post["pred_boxes"].shape) == (N, DIM * 2)
    assert tuple(preds_post["pred_probs"].shape) == (N, NUM_CLASSES)


@pytest.mark.parametrize("module_cls", [RoIBoxHead])
def test_head_specific(module_cls: Type[RoIHead], classifier, regressor_specific, coder):
    head: RoIHead = module_cls(
        classifier=classifier,
        regressor=regressor_specific,
        coder=coder,
    )
    roi_map = torch.zeros(INPUT_SIZE_TENSOR, dtype=torch.float)
    preds = head(roi_map)

    assert "box_deltas" in preds
    assert "box_logits" in preds
    assert tuple(preds["box_deltas"].shape) == (N, DIM * 2 * NUM_CLASSES)
    assert tuple(preds["box_logits"].shape) == (N, NUM_CLASSES)

    preds_post = head.postprocess_for_inference(preds, PROPOSALS)
    assert "pred_boxes" in preds_post
    assert "pred_probs" in preds_post
    assert tuple(preds_post["pred_boxes"].shape) == (N, DIM * 2 * NUM_CLASSES)
    assert tuple(preds_post["pred_probs"].shape) == (N, NUM_CLASSES)
