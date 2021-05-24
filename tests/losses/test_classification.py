import pytest

import torch
import torch.nn.functional as F

from nndet.losses.classification import (
    one_hot_smooth,
)


def test_one_hot_smooth():
    smoothing = 0.1
    # 4 classes
    targets = torch.tensor([0, 1, 2, 3])
    targets_one_hot =  torch.nn.functional.one_hot(targets, 4)
    
    targets_smooth_expected = targets_one_hot.clone().float()
    targets_smooth_expected[targets_one_hot == 1] = (1. - smoothing)
    targets_smooth_expected[targets_one_hot == 0] = smoothing / 4
    
    targets_smooth = one_hot_smooth(targets, num_classes=4, smoothing=smoothing)
    
    assert targets_smooth_expected.allclose(targets_smooth)
