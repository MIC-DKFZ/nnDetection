import torch.nn as nn

import nndet.nn.ops.norm as an

NORM_TYPES = [
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.SyncBatchNorm,
    nn.LocalResponseNorm,
    an.GroupNorm,
]

CONV_TYPES = [
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
]
