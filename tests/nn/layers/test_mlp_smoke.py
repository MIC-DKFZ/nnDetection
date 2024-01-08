import pytest
import torch

from nndet.nn.layers.mlp import ReluDropIdentityMLP, ReluMLP

INPUT_SHAPE = (128, 2, 64)  # [n_feature, bs, embed_dim]
FFN_DIM = INPUT_SHAPE[-1] * 3


TEST_CASES = [
    (
        ReluDropIdentityMLP(
            embed_dim=INPUT_SHAPE[-1],
            feedforward_dim=FFN_DIM,
            num_layers=2,
        ),
        INPUT_SHAPE,
    ),
    (
        ReluDropIdentityMLP(
            embed_dim=INPUT_SHAPE[-1],
            feedforward_dim=FFN_DIM,
            num_layers=3,
        ),
        INPUT_SHAPE,
    ),
    (
        ReluDropIdentityMLP(
            embed_dim=INPUT_SHAPE[-1],
            feedforward_dim=FFN_DIM,
            num_layers=2,
            ffn_drop=0.5,
        ),
        INPUT_SHAPE,
    ),
    (
        ReluDropIdentityMLP(
            embed_dim=INPUT_SHAPE[-1],
            feedforward_dim=FFN_DIM,
            num_layers=2,
            fc_bias=False,
        ),
        INPUT_SHAPE,
    ),
    (
        ReluMLP(
            embed_dim=INPUT_SHAPE[-1],
            feedforward_dim=FFN_DIM,
            num_layers=2,
            output_dim=32,
        ),
        (INPUT_SHAPE[0], INPUT_SHAPE[1], 32),
    ),
]


@pytest.mark.parametrize("module,expected_output_shape", TEST_CASES)
@pytest.mark.parametrize("use_identity", [True, False])
def test_mlp_smoke(module, expected_output_shape, use_identity):
    torch.manual_seed(0)

    input = torch.rand(INPUT_SHAPE)
    if use_identity:
        identity = torch.rand(expected_output_shape)
        output = module(input, identity=identity)
    else:
        output = module(input)

    assert tuple(output.shape) == expected_output_shape


def test_relu_drop_mlp_struct_nl2():
    module = ReluDropIdentityMLP(
        embed_dim=INPUT_SHAPE[-1],
        feedforward_dim=FFN_DIM,
        num_layers=2,
        ffn_drop=0.5,
    )
    assert isinstance(module.layers[0][0], torch.nn.Linear)
    assert isinstance(module.layers[0][1], torch.nn.ReLU)
    assert isinstance(module.layers[0][2], torch.nn.Dropout)
    assert module.layers[0][2].p == 0.5
    assert len(module.layers) == 3

    assert isinstance(module.layers[1], torch.nn.Linear)
    assert isinstance(module.layers[2], torch.nn.Dropout)
