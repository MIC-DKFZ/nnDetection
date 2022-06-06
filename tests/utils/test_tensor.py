import numpy as np
import pytest
import torch

from nndet.utils.tensor import ensure_min_float32, ensure_min_float32_np

TORCH_TEST_CASES = [
    (torch.float16, torch.float32),
    (torch.float32, torch.float32),
    (torch.float64, torch.float64),
]

NUMPY_TEST_CASES = [
    (np.float16, np.float64),
    (np.float32, np.float32),
    (np.float64, np.float64),
]


@pytest.mark.parametrize("input_dtype,output_dtype", TORCH_TEST_CASES)
def test_ensure_min_float(input_dtype, output_dtype):
    tensor = torch.tensor([1000, 1000, 1000], dtype=input_dtype)
    tensor_out = ensure_min_float32(tensor)
    assert tensor_out.dtype == output_dtype


@pytest.mark.parametrize("input_dtype,output_dtype", NUMPY_TEST_CASES)
def test_ensure_min_float_np(input_dtype, output_dtype):
    tensor = np.array([1000, 1000, 1000], dtype=input_dtype)
    tensor_out = ensure_min_float32_np(tensor)
    assert tensor_out.dtype == output_dtype
