import torch

from nndet.losses.ops import one_hot_smooth_last


def test_one_hot_smooth_last():
    smoothing = 0.1
    # 4 classes
    targets = torch.tensor([0, 1, 2, 3])
    targets_one_hot = torch.nn.functional.one_hot(targets, 4)

    targets_smooth_expected = targets_one_hot.clone().float()
    targets_smooth_expected[targets_one_hot == 1] = 1.0 - smoothing
    targets_smooth_expected[targets_one_hot == 0] = smoothing / 4

    targets_smooth = one_hot_smooth_last(targets, num_classes=4, smoothing=smoothing)

    assert targets_smooth_expected.allclose(targets_smooth)
