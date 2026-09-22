import torch

BACKBONE_PRESETS = {"ResEncL":{
            "n_stages":6,
            "features_per_stage":[32, 64, 128, 256, 320, 320],
            "conv_op": "torch.nn.modules.conv.Conv3d",
            "kernel_sizes":[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3],[3, 3, 3]],
            "strides":[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            "n_blocks_per_stage":[1, 3, 4, 6, 6, 6],
            "conv_bias":True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
            "norm_op_kwargs":{"eps": 1e-5, "affine": True},
            "nonlin": "torch.nn.modules.activation.LeakyReLU",
            "nonlin_kwargs":{"inplace": True}}}