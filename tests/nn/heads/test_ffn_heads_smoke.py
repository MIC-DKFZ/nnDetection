import pytest
import torch

from nndet.nn.heads.classifier.ffn import (
    BCEFFNClassifier,
    CEFFNClassifier,
    FocalFFNClassifier,
)
from nndet.nn.heads.regressor.ffn import (
    GIoUFFNRegressor,
    L1FFNRegressor,
    L1GIoUFFNRegressor,
)
from nndet.nn.layers.linear import LayerLinearReluDrop

INPUT_SIZE_TENSOR = (10, 16, 4, 4, 4)

IN_CHANNELS = 16
NUM_CLASSES = 2
DIM = 3

EXAMPLE_CONFIG = {
    "linear": LayerLinearReluDrop,
    "in_channels": 16,
    "internal_channels": 32,
    "num_layers": 2,
    "add_norm": False,
}


TEST_CASES_CLS = [
    (
        CEFFNClassifier(**EXAMPLE_CONFIG, num_classes=NUM_CLASSES),  # module
        torch.zeros(6, 4, 24, IN_CHANNELS),  # input
        torch.ones(4, 24),  # targets
        (6, 4, 24, NUM_CLASSES + 1),  # expected logits shape
        (4, 24, NUM_CLASSES),  # expected postprocess shape
    ),
    (
        BCEFFNClassifier(**EXAMPLE_CONFIG, num_classes=NUM_CLASSES),  # module
        torch.zeros(6, 4, 24, IN_CHANNELS),  # input
        torch.ones(4, 24),  # targets
        (6, 4, 24, NUM_CLASSES),  # expected logits shape
        (4, 24, NUM_CLASSES),  # expected postprocess shape
    ),
    (
        FocalFFNClassifier(**EXAMPLE_CONFIG, num_classes=NUM_CLASSES),  # module
        torch.zeros(6, 4, 24, IN_CHANNELS),  # input
        torch.ones(4, 24),  # targets
        (6, 4, 24, NUM_CLASSES),  # expected logits shape
        (4, 24, NUM_CLASSES),  # expected postprocess shape
    ),
]

TEST_CASES_REG = [
    (
        L1FFNRegressor(**EXAMPLE_CONFIG, dim=DIM),  # module
        torch.zeros(6, 4, 24, IN_CHANNELS),  # input
        torch.ones(4, 24, DIM * 2),  # targets
        (6, 4, 24, DIM * 2),  # expected logits shape
    ),
    (
        GIoUFFNRegressor(**EXAMPLE_CONFIG, dim=DIM),  # module
        torch.zeros(6, 4, 24, IN_CHANNELS),  # input
        torch.ones(4, 24, DIM * 2),  # targets
        (6, 4, 24, DIM * 2),  # expected logits shape
    ),
    (
        L1GIoUFFNRegressor(**EXAMPLE_CONFIG, dim=DIM),  # module
        torch.zeros(6, 4, 24, IN_CHANNELS),  # input
        torch.ones(4, 24, DIM * 2),  # targets
        (6, 4, 24, DIM * 2),  # expected logits shape
    ),
]


@pytest.mark.parametrize("module,inp,outp,exp_logits_shape,exp_prob_shape", TEST_CASES_CLS)
def test_ffn_cls_head_smoke(module, inp, outp, exp_logits_shape, exp_prob_shape):
    pred_logits = module(inp)
    assert pred_logits.shape == exp_logits_shape

    # compute loss for one head
    # logits [B, R, C]; targets [B, R]
    loss = module.compute_loss(pred_logits[-1].permute(0, 2, 1), outp)
    sum(loss.values()).backward()

    # last head output will be passsed to postprocess logits
    pred_probs = module.postprocess_logits(pred_logits[-1])
    assert pred_probs.shape == exp_prob_shape


@pytest.mark.parametrize("module,inp,outp,exp_logits_shape", TEST_CASES_REG)
def test_ffn_reg_head_smoke(module, inp, outp, exp_logits_shape):
    pred_logits = module(inp)
    assert pred_logits.shape == exp_logits_shape

    # compute loss for one head
    # logits [B, R, C]; targets [B, R]
    loss = module.compute_loss(
        preds=pred_logits[-1],
        targets=outp,
        pred_boxes=pred_logits[-1],
        target_boxes=outp,
    )
    sum(loss.values()).backward()
