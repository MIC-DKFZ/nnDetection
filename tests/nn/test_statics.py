import pytest
import torch

from nndet.nn.backbone.statics.resnet import ResNet10, ResNet18, ResNet34, ResNet50
from nndet.nn.layers.conv import ConvInstanceLReLU
from nndet.nn.layers.wrapper import Generator
from nndet.nn.neck.fpn import FPN, UpFPN

TEST_CONFIGS = [
    {
        "in_channels": 1,
        "conv_kernels": [3, 3, 3, 3, 3],
        "fpn_channels": 32,
        "decoder_levels": [2, 3, 4],
        "fpn_channels": 32,
        "expected_patch_size": (1, 64, 64, 64),
        "expected_shapes": [(32, 16, 16, 16), (32, 8, 8, 8), (32, 4, 4, 4)],
    },
    {
        "in_channels": 1,
        "conv_kernels": [3, 3, 3, 3, 3, 3],
        "fpn_channels": 32,
        "decoder_levels": [2, 3, 4, 5],
        "fpn_channels": 32,
        "expected_patch_size": (1, 64, 64, 64),
        "expected_shapes": [
            (32, 16, 16, 16),
            (32, 8, 8, 8),
            (32, 4, 4, 4),
            (32, 2, 2, 2),
        ],
    },
]


TEST_CASES_FPN = [
    # backbone tests
    {
        "requires": "monai",
        "backbone_cls": ResNet10,
        "backbone_kwargs": {},
        "neck_cls": FPN,
        "neck_kwargs": {},
    },
    {
        "requires": "monai",
        "backbone_cls": ResNet18,
        "backbone_kwargs": {},
        "neck_cls": FPN,
        "neck_kwargs": {},
    },
    {
        "requires": "monai",
        "backbone_cls": ResNet34,
        "backbone_kwargs": {},
        "neck_cls": FPN,
        "neck_kwargs": {},
    },
    {
        "requires": "monai",
        "backbone_cls": ResNet50,
        "backbone_kwargs": {},
        "neck_cls": FPN,
        "neck_kwargs": {},
    },
]


TEST_CASES_UFPN = [
    # neck tests
    {
        "requires": "monai",
        "backbone_cls": ResNet10,
        "backbone_kwargs": {},
        "neck_cls": UpFPN,
        "neck_kwargs": {"upsampling_mode": "transpose"},
    },
    {
        "requires": "monai",
        "backbone_cls": ResNet34,
        "backbone_kwargs": {},
        "neck_cls": UpFPN,
        "neck_kwargs": {"upsampling_mode": "transpose"},
    },
]


@pytest.mark.parametrize("network_cfg", TEST_CASES_FPN)
@pytest.mark.parametrize("test_cfg", TEST_CONFIGS)
def test_fpn_like(network_cfg, test_cfg):
    if network_cfg["backbone_cls"] is None:
        pytest.skip(f"Test requires {network_cfg['requires']} to be installed")

    expected_patch_size = test_cfg["expected_patch_size"]
    expected_shapes = test_cfg["expected_shapes"]

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

    print([type(no) for no in neck_output])

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
    if network_cfg["backbone_cls"] is None:
        pytest.skip(f"Test requires {network_cfg['requires']} to be installed")

    expected_patch_size = test_cfg["expected_patch_size"]
    expected_shapes = test_cfg["expected_shapes"]

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
