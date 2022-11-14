# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from nndet.nn.heads.abstract import RoIConv1x1View
from nndet.nn.heads.classifier.roi import (
    BCEConvRoIClassifier,
    BCEFCRoIClassifier,
    CEConvRoIClassifier,
    CEFCRoIClassifier,
)
from nndet.nn.heads.regressor.roi import (
    GIoUConvRoIRegressor,
    GIoUFCRoIRegressor,
    L1ConvRoIRegressor,
    L1FCRoIRegressor,
)
from nndet.nn.layers.conv import ConvInstanceRelu
from nndet.nn.layers.wrapper import Generator

NUM_CLASSES = 2
DIM = 3

N = 8

INPUT_SIZE_TENSOR = (N, 16, 4, 4, 4)
INPUT_SIZE_CONFIG = (4, 4, 4)
TARGET_SIZE_CLS = N
TARGET_SIZE_REG = (N, DIM * 2)

EXAMPLE_CONFIG = {
    "conv": Generator(ConvInstanceRelu, 3),
    "in_channels": 16,
    "internal_channels": 32,
    "num_convs": 1,
    "add_norm": False,
    "input_size": INPUT_SIZE_CONFIG,
}

TEST_CASES_UTIL = [
    # RoI Util
    (RoIConv1x1View(3), INPUT_SIZE_TENSOR, (10, 1024, 1, 1, 1)),
]

TEST_CASES_CLS = [
    # RoI Classifier Tests
    (
        BCEConvRoIClassifier(**EXAMPLE_CONFIG, num_classes=NUM_CLASSES),  # module
        torch.zeros(INPUT_SIZE_TENSOR),  # input
        torch.ones(TARGET_SIZE_CLS),  # target
        (N, NUM_CLASSES),  # logits shape
        (N, NUM_CLASSES),  # prob shape
    ),
    (
        BCEFCRoIClassifier(**EXAMPLE_CONFIG, num_classes=NUM_CLASSES),  # module
        torch.zeros(INPUT_SIZE_TENSOR),  # input
        torch.ones(TARGET_SIZE_CLS),  # target
        (N, NUM_CLASSES),  # logits shape
        (N, NUM_CLASSES),  # prob shape
    ),
    (
        CEConvRoIClassifier(**EXAMPLE_CONFIG, num_classes=NUM_CLASSES),  # module
        torch.zeros(INPUT_SIZE_TENSOR),  # input
        torch.ones(TARGET_SIZE_CLS),  # target
        (N, NUM_CLASSES + 1),  # logits shape
        (N, NUM_CLASSES),  # prob shape
    ),
    (
        CEFCRoIClassifier(**EXAMPLE_CONFIG, num_classes=NUM_CLASSES),  # module
        torch.zeros(INPUT_SIZE_TENSOR),  # input
        torch.ones(TARGET_SIZE_CLS),  # target
        (N, NUM_CLASSES + 1),  # logits shape
        (N, NUM_CLASSES),  # prob shape
    ),
]

TEST_CASES_REG = [
    # RoI Regressor Tests
    (
        L1ConvRoIRegressor(**EXAMPLE_CONFIG),
        torch.zeros(INPUT_SIZE_TENSOR),  # input
        torch.ones(TARGET_SIZE_REG),  # target
        (N, DIM * 2),  # logits shape
    ),
    (
        L1FCRoIRegressor(**EXAMPLE_CONFIG),
        torch.zeros(INPUT_SIZE_TENSOR),  # input
        torch.ones(TARGET_SIZE_REG),  # target
        (N, DIM * 2),  # logits shape
    ),
    (
        GIoUConvRoIRegressor(**EXAMPLE_CONFIG),
        torch.zeros(INPUT_SIZE_TENSOR),  # input
        torch.ones(TARGET_SIZE_REG),  # target
        (N, DIM * 2),  # logits shape
    ),
    (
        GIoUFCRoIRegressor(**EXAMPLE_CONFIG),
        torch.zeros(INPUT_SIZE_TENSOR),  # input
        torch.ones(TARGET_SIZE_REG),  # target
        (N, DIM * 2),  # logits shape
    ),
]


@pytest.mark.parametrize("module,input_size,expected_shape", TEST_CASES_UTIL)
def test_roi_util_smoke(module, input_size, expected_shape):
    inp = torch.zeros(input_size)
    outp = module(inp)
    assert outp.shape == expected_shape


@pytest.mark.parametrize("module,inp,target,exp_logits_shape,exp_probs_shape", TEST_CASES_CLS)
def test_roi_cls_head_smoke(module, inp, target, exp_logits_shape, exp_probs_shape):
    pred_logits = module(inp)
    assert tuple(pred_logits.shape) == exp_logits_shape

    loss = module.compute_loss(pred_logits, target)
    loss.backward()

    pred_probs = module.logits_to_probs(pred_logits)
    assert tuple(pred_probs.shape) == exp_probs_shape


@pytest.mark.parametrize("module,inp,target,exp_shape", TEST_CASES_REG)
def test_roi_reg_head_smoke(module, inp, target, exp_shape):
    pred_logits = module(inp)
    assert tuple(pred_logits.shape) == exp_shape

    loss = module.compute_loss(pred_logits, target)
    loss.backward()
