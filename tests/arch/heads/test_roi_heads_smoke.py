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
from nndet.nn.layers.conv import ConvInstanceRelu, Generator

INPUT_SIZE_TENSOR = (10, 16, 4, 4, 4)
INPUT_SIZE_CONFIG = (4, 4, 4)

EXAMPLE_CONFIG = {
    "conv": Generator(ConvInstanceRelu, 3),
    "in_channels": 16,
    "internal_channels": 32,
    "num_convs": 1,
    "add_norm": False,
    "input_size": INPUT_SIZE_CONFIG,
}

TEST_CASES = [
    # RoI Util
    (RoIConv1x1View(3), INPUT_SIZE_TENSOR, (10, 1024, 1, 1, 1)),
    # RoI Regressor Tests
    (L1ConvRoIRegressor(**EXAMPLE_CONFIG), INPUT_SIZE_TENSOR, (10, 6)),
    (L1FCRoIRegressor(**EXAMPLE_CONFIG), INPUT_SIZE_TENSOR, (10, 6)),
    (GIoUConvRoIRegressor(**EXAMPLE_CONFIG), INPUT_SIZE_TENSOR, (10, 6)),
    (GIoUFCRoIRegressor(**EXAMPLE_CONFIG), INPUT_SIZE_TENSOR, (10, 6)),
    # RoI Classifier Tests
    (BCEConvRoIClassifier(**EXAMPLE_CONFIG, num_classes=1), INPUT_SIZE_TENSOR, (10, 1)),
    (BCEFCRoIClassifier(**EXAMPLE_CONFIG, num_classes=1), INPUT_SIZE_TENSOR, (10, 1)),
    (CEConvRoIClassifier(**EXAMPLE_CONFIG, num_classes=1), INPUT_SIZE_TENSOR, (10, 2)),
    (CEFCRoIClassifier(**EXAMPLE_CONFIG, num_classes=1), INPUT_SIZE_TENSOR, (10, 2)),
]


@pytest.mark.parametrize("module,input_size,expected_shape", TEST_CASES)
def test_roi_head_smoke(module, input_size, expected_shape):
    inp = torch.zeros(input_size)
    outp = module(inp)
    assert outp.shape == expected_shape
