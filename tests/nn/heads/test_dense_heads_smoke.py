# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from nndet.nn.heads.classifier.dense import (
    AsymmetricFocalClassifier,
    BCECLassifier,
    CEClassifier,
    FocalClassifier,
    FullyConntectedBCECLassifier,
    Poly1BCECLassifier,
    Poly1FocalClassifier,
)
from nndet.nn.heads.regressor.dense import (
    DIoURegressor,
    DualRegressor,
    GIoUPRegressor,
    GIoURegressor,
    L1Regressor,
)
from nndet.nn.layers.conv import ConvInstanceRelu
from nndet.nn.layers.wrapper import Generator

NUM_CLASSES = 2
DIM = 3

N = 8

ANCHORS_PER_POS = 9
TOTAL_ANCHORS = 4 * 4 * 4 * ANCHORS_PER_POS
NUM_LEVELS = 3

INPUT_SIZE_TENSOR = (N, 16, 4, 4, 4)
TARGET_SIZE_CLS = N * TOTAL_ANCHORS
TARGET_SIZE_REG = (N * TOTAL_ANCHORS, DIM * 2)

EXAMPLE_CONFIG = {
    "conv": Generator(ConvInstanceRelu, 3),
    "in_channels": 16,
    "internal_channels": 32,
    "num_convs": 1,
    "add_norm": True,
    "num_classes": NUM_CLASSES,
}

TEST_CASES_CLS = [
    # Dense Classifier Tests
    (
        BCECLassifier(
            **EXAMPLE_CONFIG,
            anchors_per_pos=ANCHORS_PER_POS,
            num_levels=3,
        ),  # module
        torch.zeros(INPUT_SIZE_TENSOR),  # input
        torch.ones(TARGET_SIZE_CLS),  # target
        (N, TOTAL_ANCHORS, NUM_CLASSES),  # logits shape
        (N * TOTAL_ANCHORS, NUM_CLASSES),  # prob shape
    ),
    (
        CEClassifier(
            **EXAMPLE_CONFIG,
            anchors_per_pos=ANCHORS_PER_POS,
            num_levels=3,
        ),  # module
        torch.zeros(INPUT_SIZE_TENSOR),  # input
        torch.ones(TARGET_SIZE_CLS),  # target
        (N, TOTAL_ANCHORS, NUM_CLASSES + 1),  # logits shape
        (N * TOTAL_ANCHORS, NUM_CLASSES),  # prob shape
    ),
    (
        FocalClassifier(
            **EXAMPLE_CONFIG,
            anchors_per_pos=ANCHORS_PER_POS,
            num_levels=3,
        ),  # module
        torch.zeros(INPUT_SIZE_TENSOR),  # input
        torch.ones(TARGET_SIZE_CLS),  # target
        (N, TOTAL_ANCHORS, NUM_CLASSES),  # logits shape
        (N * TOTAL_ANCHORS, NUM_CLASSES),  # prob shape
    ),
    (
        AsymmetricFocalClassifier(
            **EXAMPLE_CONFIG,
            anchors_per_pos=ANCHORS_PER_POS,
            num_levels=3,
        ),  # module
        torch.zeros(INPUT_SIZE_TENSOR),  # input
        torch.ones(TARGET_SIZE_CLS),  # target
        (N, TOTAL_ANCHORS, NUM_CLASSES),  # logits shape
        (N * TOTAL_ANCHORS, NUM_CLASSES),  # prob shape
    ),
    (
        Poly1BCECLassifier(
            **EXAMPLE_CONFIG,
            anchors_per_pos=ANCHORS_PER_POS,
            num_levels=3,
        ),  # module
        torch.zeros(INPUT_SIZE_TENSOR),  # input
        torch.ones(TARGET_SIZE_CLS),  # target
        (N, TOTAL_ANCHORS, NUM_CLASSES),  # logits shape
        (N * TOTAL_ANCHORS, NUM_CLASSES),  # prob shape
    ),
    (
        Poly1FocalClassifier(
            **EXAMPLE_CONFIG,
            anchors_per_pos=ANCHORS_PER_POS,
            num_levels=3,
        ),  # module
        torch.zeros(INPUT_SIZE_TENSOR),  # input
        torch.ones(TARGET_SIZE_CLS),  # target
        (N, TOTAL_ANCHORS, NUM_CLASSES),  # logits shape
        (N * TOTAL_ANCHORS, NUM_CLASSES),  # prob shape
    ),
    (
        FullyConntectedBCECLassifier(
            **EXAMPLE_CONFIG,
            anchors_per_pos=ANCHORS_PER_POS,
            num_levels=3,
        ),  # module
        torch.zeros(INPUT_SIZE_TENSOR),  # input
        torch.ones(TARGET_SIZE_CLS),  # target
        (N, TOTAL_ANCHORS, NUM_CLASSES),  # logits shape
        (N * TOTAL_ANCHORS, NUM_CLASSES),  # prob shape
    ),
]

TEST_CASES_REG = [
    # Dense Regressor Tests
    (
        L1Regressor(
            **EXAMPLE_CONFIG,
            anchors_per_pos=ANCHORS_PER_POS,
            num_levels=3,
        ),  # module
        torch.zeros(INPUT_SIZE_TENSOR),  # input
        torch.ones(TARGET_SIZE_REG),  # target
        (N, TOTAL_ANCHORS, DIM * 2),  # logits shape
    ),
    (
        L1Regressor(
            **EXAMPLE_CONFIG,
            anchors_per_pos=ANCHORS_PER_POS,
            num_levels=3,
            learn_scale=True,
        ),  # module
        torch.zeros(INPUT_SIZE_TENSOR),  # input
        torch.ones(TARGET_SIZE_REG),  # target
        (N, TOTAL_ANCHORS, DIM * 2),  # logits shape
    ),
    (
        GIoURegressor(
            **EXAMPLE_CONFIG,
            anchors_per_pos=ANCHORS_PER_POS,
            num_levels=3,
        ),  # module
        torch.zeros(INPUT_SIZE_TENSOR),  # input
        torch.ones(TARGET_SIZE_REG),  # target
        (N, TOTAL_ANCHORS, DIM * 2),  # logits shape
    ),
    (
        GIoUPRegressor(
            **EXAMPLE_CONFIG,
            anchors_per_pos=ANCHORS_PER_POS,
            num_levels=3,
        ),  # module
        torch.zeros(INPUT_SIZE_TENSOR),  # input
        torch.ones(TARGET_SIZE_REG),  # target
        (N, TOTAL_ANCHORS, DIM * 2),  # logits shape
    ),
    (
        DIoURegressor(
            **EXAMPLE_CONFIG,
            anchors_per_pos=ANCHORS_PER_POS,
            num_levels=3,
        ),  # module
        torch.zeros(INPUT_SIZE_TENSOR),  # input
        torch.ones(TARGET_SIZE_REG),  # target
        (N, TOTAL_ANCHORS, DIM * 2),  # logits shape
    ),
]

TEST_CASES_DUAL_REG = [
    # Dense Dual Regressor Tests
    (
        DualRegressor(
            **EXAMPLE_CONFIG,
            anchors_per_pos=ANCHORS_PER_POS,
            num_levels=3,
        ),  # module
        torch.zeros(INPUT_SIZE_TENSOR),  # input
        torch.ones(TARGET_SIZE_REG),  # target
        (N, TOTAL_ANCHORS, DIM * 2),  # logits shape
    ),
]


@pytest.mark.parametrize("module,inp,target,exp_logits_shape,exp_probs_shape", TEST_CASES_CLS)
def test_dense_cls_head_smoke(module, inp, target, exp_logits_shape, exp_probs_shape):
    pred_logits = module(inp, level=0)
    assert tuple(pred_logits.shape) == exp_logits_shape

    pred_logits_flattened = pred_logits.flatten(0, -2)
    loss = module.compute_loss(pred_logits_flattened, target)
    loss.backward()

    pred_probs_flattened = module.logits_to_probs(pred_logits_flattened)
    assert tuple(pred_probs_flattened.shape) == exp_probs_shape


@pytest.mark.parametrize("module,inp,target,exp_shape", TEST_CASES_REG)
def test_dense_reg_head_smoke(module, inp, target, exp_shape):
    pred_logits = module(inp, level=0)
    assert tuple(pred_logits.shape) == exp_shape

    pred_logits_flattened = pred_logits.reshape(-1, DIM * 2)
    loss = module.compute_loss(pred_logits_flattened, target)
    loss.backward()


@pytest.mark.parametrize("module,inp,target,exp_shape", TEST_CASES_DUAL_REG)
def test_dense_dual_reg_head_smoke(module, inp, target, exp_shape):
    pred_logits = module(inp, level=0)
    assert tuple(pred_logits.shape) == exp_shape

    pred_logits_flattened = pred_logits.reshape(-1, DIM * 2)
    loss = module.compute_loss(
        pred_deltas=pred_logits_flattened,
        target_deltas=target,
        pred_boxes=pred_logits_flattened,
        target_boxes=target,
    )
    loss.backward()
