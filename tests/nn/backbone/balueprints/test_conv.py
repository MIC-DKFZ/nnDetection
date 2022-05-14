import pytest
import torch

from nndet.nn.backbone.blueprints.conv import ConvBackbone
from nndet.nn.layers.conv import ConvInstanceLReLU
from nndet.nn.layers.wrapper import Generator

CASES = [
    ({"pooling_mode": "conv_kernel"}, {}),
    ({"pooling_mode": "conv_stride"}, {}),
    ({"pooling_mode": "max_kernel"}, {}),
    ({"pooling_mode": "max_stride"}, {}),
    ({"pooling_mode": "avg_kernel"}, {}),
    ({"pooling_mode": "avg_stride"}, {}),
    ({"num_conv": 3}, {}),
    ({"max_channels": 128}, {}),
]


class TestConvBackbone:
    @pytest.mark.parametrize("backbone_cfg,plan_arch", CASES)
    def test_smoke_forward(self, backbone_cfg: dict, plan_arch: dict):
        backbone = ConvBackbone.from_config_plan(
            conv=Generator(ConvInstanceLReLU, dim=2),
            backbone_cfg={
                **backbone_cfg,
            },
            plan_arch={
                "conv_kernels": [3, 3, 3, 3, 3, 3],
                "strides": [2, 2, 2, 2, 2],
                "in_channels": 1,
                "start_channels": 8,
                "max_channels": 256,
                **plan_arch,
            },
        )
        out = backbone(torch.zeros(1, 1, 128, 128))
        if "max_channels" in backbone_cfg:
            assert tuple(out[-1].shape) == (1, backbone_cfg["max_channels"], 4, 4)
        else:
            assert tuple(out[-1].shape) == (1, 256, 4, 4)
        assert len(out) == 6
