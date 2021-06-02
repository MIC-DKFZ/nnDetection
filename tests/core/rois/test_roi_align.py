import pytest

import torch
from torchvision.ops.roi_align import roi_align as _roi_align


def test_roi_align():
    boxes = torch.tensor([[4., 4., 6., 6.]])
    # boxes = torch.tensor([[2., 2., 3., 3.]])
    # fmap = torch.zeros(1, 1, 10, 10)
    # fmap[:, :, 4:8, 4:8] = 1
    
    fmap = torch.zeros(1, 1, 5, 5)
    fmap[:, :, 2:4, 2:4] = 1

    pooled_fmap = _roi_align(fmap, [boxes],
                             output_size=(3, 3),
                             spatial_scale=0.5,
                             # sampling_ratio=2,
                             )

    print(pooled_fmap)
    c =1
