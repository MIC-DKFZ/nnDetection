# Test Configs
# 5 Levels Iso
# 5 Levels First Aniso
# 5 Levels Two Aniso
# 4 Levels Iso Neck Starts at 1

# expected results (will be popped for test)

import pytest
import torch

from nndet.nn.backbone.blueprints.conv import ConvBackbone
from nndet.nn.layers.conv import ConvInstanceLReLU
from nndet.nn.layers.wrapper import Generator
from nndet.nn.neck.fpn import FPN, UFPN, UpFPN

TEST_CONFIGS = [
    {
        "conv_kernels": [3, 3, 3, 3, 3],
        "strides": [2, 2, 2, 2],
        "in_channels": 1,
        "start_channels": 4,
        "max_channels": 32,
        "decoder_levels": [2, 3, 4],
        "fpn_channels": 32,
        "expected_patch_size": (1, 64, 64, 64),
        "expected_shapes": [(32, 16, 16, 16), (32, 8, 8, 8), (32, 4, 4, 4)],
    },
    {
        "conv_kernels": [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
        "strides": [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
        "in_channels": 1,
        "start_channels": 4,
        "max_channels": 32,
        "decoder_levels": [2, 3, 4],
        "fpn_channels": 32,
        "expected_patch_size": (1, 64, 64, 64),
        "expected_shapes": [(32, 16, 16, 16), (32, 8, 8, 8), (32, 4, 4, 4)],
    },
    {
        "conv_kernels": [(1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
        "strides": [(1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
        "in_channels": 1,
        "start_channels": 4,
        "max_channels": 32,
        "decoder_levels": [2, 3, 4],
        "fpn_channels": 32,
        "expected_patch_size": (1, 64, 64, 64),
        "expected_shapes": [(32, 32, 16, 16), (32, 16, 8, 8), (32, 8, 4, 4)],
    },
    {
        "conv_kernels": [(1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
        "strides": [(1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2)],
        "in_channels": 1,
        "start_channels": 4,
        "max_channels": 32,
        "decoder_levels": [2, 3, 4],
        "fpn_channels": 32,
        "expected_patch_size": (1, 64, 64, 64),
        "expected_shapes": [(32, 64, 16, 16), (32, 32, 8, 8), (32, 16, 4, 4)],
    },
    {
        "conv_kernels": [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
        "strides": [(2, 2, 2), (2, 2, 2), (2, 2, 2)],
        "in_channels": 1,
        "start_channels": 4,
        "max_channels": 32,
        "decoder_levels": [1, 2, 3],
        "fpn_channels": 32,
        "expected_patch_size": (1, 64, 64, 64),
        "expected_shapes": [(32, 32, 32, 32), (32, 16, 16, 16), (32, 8, 8, 8)],
    },
]


TEST_CASES_FPN = [
    # backbone tests
    {
        "backbone_cls": ConvBackbone,
        "backbone_kwargs": {"num_conv": 2, "pooling_mode": "conv_kernel"},
        "neck_cls": FPN,
        "neck_kwargs": {},
    },
    {
        "backbone_cls": ConvBackbone,
        "backbone_kwargs": {"num_conv": 3, "pooling_mode": "conv_kernel"},
        "neck_cls": FPN,
        "neck_kwargs": {},
    },
    {
        "backbone_cls": ConvBackbone,
        "backbone_kwargs": {"num_conv": 2, "pooling_mode": "conv_stride"},
        "neck_cls": FPN,
        "neck_kwargs": {},
    },
    {
        "backbone_cls": ConvBackbone,
        "backbone_kwargs": {"num_conv": 2, "pooling_mode": "max_kernel"},
        "neck_cls": FPN,
        "neck_kwargs": {},
    },
    {
        "backbone_cls": ConvBackbone,
        "backbone_kwargs": {"num_conv": 2, "pooling_mode": "avg_kernel"},
        "neck_cls": FPN,
        "neck_kwargs": {},
    },
    # neck tests
    {
        "backbone_cls": ConvBackbone,
        "backbone_kwargs": {"num_conv": 2, "pooling_mode": "conv_kernel"},
        "neck_cls": FPN,
        "neck_kwargs": {"upsampling_mode": "transpose"},
    },
    {
        "backbone_cls": ConvBackbone,
        "backbone_kwargs": {"num_conv": 2, "pooling_mode": "conv_kernel"},
        "neck_cls": FPN,
        "neck_kwargs": {
            "upsampling_mode": "transpose",
            "num_lateral": 2,
            "norm_lateral": True,
            "activation_lateral": True,
        },
    },
    {
        "backbone_cls": ConvBackbone,
        "backbone_kwargs": {"num_conv": 2, "pooling_mode": "conv_kernel"},
        "neck_cls": FPN,
        "neck_kwargs": {
            "upsampling_mode": "transpose",
            "num_out": 2,
            "norm_out": True,
            "activation_out": True,
        },
    },
    {
        "backbone_cls": ConvBackbone,
        "backbone_kwargs": {"num_conv": 2, "pooling_mode": "conv_kernel"},
        "neck_cls": FPN,
        "neck_kwargs": {
            "upsampling_mode": "transpose",
            "num_fusion": 2,
            "norm_fusion": True,
            "activation_fusion": True,
        },
    },
]


TEST_CASES_UFPN = [
    # neck tests
    {
        "backbone_cls": ConvBackbone,
        "backbone_kwargs": {"num_conv": 2, "pooling_mode": "conv_kernel"},
        "neck_cls": UFPN,
        "neck_kwargs": {"upsampling_mode": "transpose"},
    },
    {
        "backbone_cls": ConvBackbone,
        "backbone_kwargs": {"num_conv": 2, "pooling_mode": "conv_kernel"},
        "neck_cls": UFPN,
        "neck_kwargs": {"upsampling_mode": "linear"},
    },
    {
        "backbone_cls": ConvBackbone,
        "backbone_kwargs": {"num_conv": 2, "pooling_mode": "conv_kernel"},
        "neck_cls": UpFPN,
        "neck_kwargs": {"upsampling_mode": "transpose"},
    },
    {
        "backbone_cls": ConvBackbone,
        "backbone_kwargs": {"num_conv": 2, "pooling_mode": "conv_kernel"},
        "neck_cls": UpFPN,
        "neck_kwargs": {"upsampling_mode": "linear"},
    },
]


@pytest.mark.parametrize("network_cfg", TEST_CASES_FPN)
@pytest.mark.parametrize("test_cfg", TEST_CONFIGS)
def test_fpn_like(network_cfg, test_cfg):
    expected_patch_size = test_cfg.pop("expected_patch_size")
    expected_shapes = test_cfg.pop("expected_shapes")

    conv = Generator(ConvInstanceLReLU, 3)
    backbone = network_cfg["backbone_cls"].from_config_plan(
        conv=conv,
        backbone_cfg=network_cfg["backbone_kwargs"],
        plan_arch=test_cfg,
    )
    decoder_levels = test_cfg["decoder_levels"]
    neck = network_cfg["neck_cls"](
        conv=conv,
        conv_kernels=test_cfg["conv_kernels"],
        relative_strides=backbone.get_relative_strides(),
        in_channels=backbone.get_channels(),
        first_decoder_level=min(decoder_levels),
        last_decoder_level=max(decoder_levels),
        fpn_out_channels=test_cfg["fpn_channels"],
        **network_cfg["neck_kwargs"],
    )

    input_tensor = torch.zeros(1, *expected_patch_size)
    backbone_output = backbone(input_tensor)
    neck_output = neck(backbone_output)

    for shape_idx, level_idx in enumerate(decoder_levels):
        assert tuple(neck_output[level_idx].shape[1:]) == expected_shapes[shape_idx]

    output_channels = neck.compute_output_channels()
    assert len(output_channels) == len(neck_output)
    for level_idx, channels in enumerate(output_channels):
        if channels is None:
            assert neck_output[level_idx] is None
        else:
            assert neck_output[level_idx].shape[1] == channels


@pytest.mark.parametrize("network_cfg", TEST_CASES_UFPN)
@pytest.mark.parametrize("test_cfg", TEST_CONFIGS)
def test_ufpn_like(network_cfg, test_cfg):
    expected_patch_size = test_cfg.pop("expected_patch_size")
    expected_shapes = test_cfg.pop("expected_shapes")

    conv = Generator(ConvInstanceLReLU, 3)
    backbone = network_cfg["backbone_cls"].from_config_plan(
        conv=conv,
        backbone_cfg=network_cfg["backbone_kwargs"],
        plan_arch=test_cfg,
    )
    decoder_levels = test_cfg["decoder_levels"]
    neck = network_cfg["neck_cls"](
        conv=conv,
        conv_kernels=test_cfg["conv_kernels"],
        relative_strides=backbone.get_relative_strides(),
        in_channels=backbone.get_channels(),
        first_decoder_level=min(decoder_levels),
        last_decoder_level=max(decoder_levels),
        fpn_out_channels=test_cfg["fpn_channels"],
        **network_cfg["neck_kwargs"],
    )

    input_tensor = torch.zeros(1, *expected_patch_size)
    backbone_output = backbone(input_tensor)
    neck_output = neck(backbone_output)

    for shape_idx, level_idx in enumerate(decoder_levels):
        assert tuple(neck_output[level_idx].shape[1:]) == expected_shapes[shape_idx]

    output_channels = neck.compute_output_channels()
    assert len(output_channels) == len(neck_output)
    for level_idx, channels in enumerate(output_channels):
        if channels is None:
            assert neck_output[level_idx] is None
        else:
            assert neck_output[level_idx].shape[1] == channels

    # check stride 1 output
    assert neck_output[0].shape[2:] == expected_patch_size[1:]
