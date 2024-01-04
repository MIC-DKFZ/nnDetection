import pytest
import torch

from nndet.nn.layers.conv.instance import ConvInstanceLReLU
from nndet.nn.layers.wrapper import Generator
from nndet.nn.neck.channel_mapper import ChannelMapper

BS = 2
IN_CHANNELS = [
    256,
    128,
    64,
    32,
]  # from smallest feature map to largest, starting a 2 and doubling in size


TEST_CONFIGS = [
    (
        2,
        {
            "mapper_cls": ChannelMapper,
            "num_in_features": 4,
            "kernel_size": 3,
            "out_channels": 32,
            "num_outs": None,
        },
        [
            (BS, 32, 16, 16),
            (BS, 32, 8, 8),
            (BS, 32, 4, 4),
            (BS, 32, 2, 2),
        ],
    ),
    (
        3,
        {
            "mapper_cls": ChannelMapper,
            "num_in_features": 4,
            "kernel_size": 3,
            "out_channels": 32,
            "num_outs": None,
        },
        [
            (BS, 32, 16, 16, 16),
            (BS, 32, 8, 8, 8),
            (BS, 32, 4, 4, 4),
            (BS, 32, 2, 2, 2),
        ],
    ),
    (
        3,
        {
            "mapper_cls": ChannelMapper,
            "num_in_features": 3,
            "kernel_size": 3,
            "out_channels": 32,
            "num_outs": None,
        },
        [
            (BS, 32, 8, 8, 8),
            (BS, 32, 4, 4, 4),
            (BS, 32, 2, 2, 2),
        ],
    ),
    (
        3,
        {
            "mapper_cls": ChannelMapper,
            "num_in_features": 3,
            "kernel_size": 3,
            "out_channels": 32,
            "num_outs": 3,
        },
        [
            (BS, 32, 8, 8, 8),
            (BS, 32, 4, 4, 4),
            (BS, 32, 2, 2, 2),
        ],
    ),
    (
        3,
        {
            "mapper_cls": ChannelMapper,
            "num_in_features": 3,
            "kernel_size": 3,
            "out_channels": 32,
            "num_outs": 5,
        },
        [
            (BS, 32, 8, 8, 8),
            (BS, 32, 4, 4, 4),
            (BS, 32, 2, 2, 2),
            (BS, 32, 2, 2, 2),
            (BS, 32, 2, 2, 2),
        ],
    ),
]


@pytest.mark.parametrize("dim,mapper_cfg,expected_output_shapes", TEST_CONFIGS)
def test_fpn_like(dim, mapper_cfg, expected_output_shapes):
    conv = Generator(ConvInstanceLReLU, dim)

    if dim == 2:
        backbone_features = [torch.rand((BS, i, 2**idx, 2**idx)) for idx, i in enumerate(IN_CHANNELS, start=1)][
            ::-1
        ]
    elif dim == 3:
        backbone_features = [
            torch.rand((BS, i, 2**idx, 2**idx, 2**idx)) for idx, i in enumerate(IN_CHANNELS, start=1)
        ][::-1]
    else:
        raise RuntimeError()

    mapper: ChannelMapper = mapper_cfg["mapper_cls"](
        conv=conv,
        in_channels=IN_CHANNELS[::-1],
        num_in_features=mapper_cfg["num_in_features"],
        kernel_size=mapper_cfg["kernel_size"],
        out_channels=mapper_cfg["out_channels"],
        num_outs=mapper_cfg["num_outs"],
        **mapper_cfg.get("kwargs", {}),
    )

    mapper_out = mapper(backbone_features)
    assert len(mapper_out) == len(expected_output_shapes)
    for i, out in enumerate(mapper_out):
        assert tuple(out.shape) == expected_output_shapes[i]
