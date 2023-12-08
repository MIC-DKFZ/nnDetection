import numpy as np
import pytest
import torch

from nndet.utils.tensor import detach_all, ensure_min_float32, ensure_min_float32_np

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


def test_detach_all_dict_tensor():
    boxes = torch.zeros((2, 6), dtype=torch.float, requires_grad=True)
    scores = torch.zeros((2,), dtype=torch.float, requires_grad=True)

    data = {"boxes": boxes, "scores": scores}
    data_detached = detach_all(data)

    assert data_detached["boxes"].requires_grad is False
    assert data_detached["scores"].requires_grad is False
    assert boxes.requires_grad is True
    assert scores.requires_grad is True


def test_detach_all_list_tensor():
    boxes = torch.zeros((2, 6), dtype=torch.float, requires_grad=True)
    scores = torch.zeros((2,), dtype=torch.float, requires_grad=True)

    data = [boxes, scores]
    data_detached = detach_all(data)

    assert data_detached[0].requires_grad is False
    assert data_detached[1].requires_grad is False
    assert boxes.requires_grad is True
    assert scores.requires_grad is True


def test_detach_all_dict_list_tensor():
    boxes = torch.zeros((2, 6), dtype=torch.float, requires_grad=True)
    scores = torch.zeros((2,), dtype=torch.float, requires_grad=True)

    data = {"boxes": [boxes], "scores": [scores]}
    data_detached = detach_all(data)

    assert data_detached["boxes"][0].requires_grad is False
    assert data_detached["scores"][0].requires_grad is False
    assert boxes.requires_grad is True
    assert scores.requires_grad is True
