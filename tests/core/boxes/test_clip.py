import pytest
import torch

import nndet.core.ops_torch as ops_torch


def test_clip_boxes_2d():
    boxes = torch.tensor([[-1, -2, 102, 200]])
    boxes = ops_torch.clip_boxes_to_image_(boxes, (100, 200))
    expected = torch.tensor([[0, 0, 100, 200]])
    assert (expected == boxes).all()


def test_clip_boxes_3d():
    boxes = torch.tensor([[-1, -2, 102, 200, -5, 30]])
    boxes = ops_torch.clip_boxes_to_image_(boxes, (100, 200, 16))
    expected = torch.tensor([[0, 0, 100, 200, 0, 16]])
    assert (expected == boxes).all()
