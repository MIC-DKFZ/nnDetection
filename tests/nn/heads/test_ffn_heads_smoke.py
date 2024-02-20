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

IN_CHANNELS = 16
NUM_CLASSES = 3
DIM = 3
NUM_DECODER_LAYERS = 2

EXAMPLE_CONFIG = {
    "linear": LayerLinearReluDrop,
    "in_channels": IN_CHANNELS,
    "internal_channels": 32,
    "num_layers": 2,
    "add_norm": False,
    "num_decoder_layers": NUM_DECODER_LAYERS,
}


TEST_CASES_CLS = [
    (
        CEFFNClassifier(**EXAMPLE_CONFIG, num_classes=NUM_CLASSES),  # module
        torch.zeros(NUM_DECODER_LAYERS, 4, 24, IN_CHANNELS),  # input
        torch.ones(4, 24),  # targets
        (NUM_DECODER_LAYERS, 4, 24, NUM_CLASSES + 1),  # expected logits shape
        (4, 24, NUM_CLASSES),  # expected postprocess shape
    ),
    (
        BCEFFNClassifier(**EXAMPLE_CONFIG, num_classes=NUM_CLASSES),  # module
        torch.zeros(NUM_DECODER_LAYERS, 4, 24, IN_CHANNELS),  # input
        torch.ones(4, 24),  # targets
        (NUM_DECODER_LAYERS, 4, 24, NUM_CLASSES),  # expected logits shape
        (4, 24, NUM_CLASSES),  # expected postprocess shape
    ),
    (
        FocalFFNClassifier(**EXAMPLE_CONFIG, num_classes=NUM_CLASSES),  # module
        torch.zeros(NUM_DECODER_LAYERS, 4, 24, IN_CHANNELS),  # input
        torch.ones(4, 24),  # targets
        (NUM_DECODER_LAYERS, 4, 24, NUM_CLASSES),  # expected logits shape
        (4, 24, NUM_CLASSES),  # expected postprocess shape
    ),
    (
        FocalFFNClassifier(**EXAMPLE_CONFIG, num_classes=NUM_CLASSES, use_encoder_mlp=True),  # module
        torch.zeros(NUM_DECODER_LAYERS, 4, 24, IN_CHANNELS),  # input
        torch.ones(4, 24),  # targets
        (NUM_DECODER_LAYERS, 4, 24, NUM_CLASSES),  # expected logits shape
        (4, 24, NUM_CLASSES),  # expected postprocess shape
    ),
]


TEST_CASES_CLS_SEP = [
    # CLS test cases with separate forward pass of each output sequence element
    (
        FocalFFNClassifier(**EXAMPLE_CONFIG, num_classes=NUM_CLASSES, share_mlp=False),  # module
        torch.zeros(NUM_DECODER_LAYERS, 4, 24, IN_CHANNELS),  # input
        torch.ones(4, 24),  # targets
        (4, 24, NUM_CLASSES),  # expected logits shape
        (4, 24, NUM_CLASSES),  # expected postprocess shape
        (4, 24, NUM_CLASSES),  # expected auxiliary shape
    ),
    # check class agnostic aux head
    (
        CEFFNClassifier(**EXAMPLE_CONFIG, num_classes=NUM_CLASSES, share_mlp=False, class_agnostic_aux=True),  # module
        torch.zeros(NUM_DECODER_LAYERS, 4, 24, IN_CHANNELS),  # input
        torch.ones(4, 24),  # targets
        (4, 24, NUM_CLASSES + 1),  # expected logits shape
        (4, 24, NUM_CLASSES),  # expected postprocess shape
        (4, 24, 2),  # expected auxiliary shape
    ),
    (
        BCEFFNClassifier(**EXAMPLE_CONFIG, num_classes=NUM_CLASSES, share_mlp=False, class_agnostic_aux=True),  # module
        torch.zeros(NUM_DECODER_LAYERS, 4, 24, IN_CHANNELS),  # input
        torch.ones(4, 24),  # targets
        (4, 24, NUM_CLASSES),  # expected logits shape
        (4, 24, NUM_CLASSES),  # expected postprocess shape
        (4, 24, 1),  # expected auxiliary shape
    ),
    (
        FocalFFNClassifier(
            **EXAMPLE_CONFIG, num_classes=NUM_CLASSES, share_mlp=False, class_agnostic_aux=True
        ),  # module
        torch.zeros(NUM_DECODER_LAYERS, 4, 24, IN_CHANNELS),  # input
        torch.ones(4, 24),  # targets
        (4, 24, NUM_CLASSES),  # expected logits shape
        (4, 24, NUM_CLASSES),  # expected postprocess shape
        (4, 24, 1),  # expected auxiliary shape
    ),
    # check share mlp (smoke)
    (
        FocalFFNClassifier(**EXAMPLE_CONFIG, num_classes=NUM_CLASSES, share_mlp=False, use_encoder_mlp=True),  # module
        torch.zeros(NUM_DECODER_LAYERS, 4, 24, IN_CHANNELS),  # input
        torch.ones(4, 24),  # targets
        (4, 24, NUM_CLASSES),  # expected logits shape
        (4, 24, NUM_CLASSES),  # expected postprocess shape
        (4, 24, NUM_CLASSES),  # expected auxiliary shape
    ),
]

TEST_CASES_REG = [
    (
        L1FFNRegressor(**EXAMPLE_CONFIG, dim=DIM),  # module
        torch.zeros(NUM_DECODER_LAYERS, 4, 24, IN_CHANNELS),  # input
        torch.ones(4, 24, DIM * 2),  # targets
        (NUM_DECODER_LAYERS, 4, 24, DIM * 2),  # expected logits shape
    ),
    (
        GIoUFFNRegressor(**EXAMPLE_CONFIG, dim=DIM),  # module
        torch.zeros(NUM_DECODER_LAYERS, 4, 24, IN_CHANNELS),  # input
        torch.ones(4, 24, DIM * 2),  # targets
        (NUM_DECODER_LAYERS, 4, 24, DIM * 2),  # expected logits shape
    ),
    (
        L1GIoUFFNRegressor(**EXAMPLE_CONFIG, dim=DIM),  # module
        torch.zeros(NUM_DECODER_LAYERS, 4, 24, IN_CHANNELS),  # input
        torch.ones(4, 24, DIM * 2),  # targets
        (NUM_DECODER_LAYERS, 4, 24, DIM * 2),  # expected logits shape
    ),
]


TEST_CASES_REG_SEP = [
    (
        L1FFNRegressor(**EXAMPLE_CONFIG, dim=DIM, share_mlp=False),  # module
        torch.zeros(NUM_DECODER_LAYERS, 4, 24, IN_CHANNELS),  # input
        torch.ones(4, 24, DIM * 2),  # targets
        (4, 24, DIM * 2),  # expected logits shape
    ),
    (
        GIoUFFNRegressor(**EXAMPLE_CONFIG, dim=DIM, share_mlp=False),  # module
        torch.zeros(NUM_DECODER_LAYERS, 4, 24, IN_CHANNELS),  # input
        torch.ones(4, 24, DIM * 2),  # targets
        (4, 24, DIM * 2),  # expected logits shape
    ),
    (
        L1GIoUFFNRegressor(**EXAMPLE_CONFIG, dim=DIM, share_mlp=False),  # module
        torch.zeros(NUM_DECODER_LAYERS, 4, 24, IN_CHANNELS),  # input
        torch.ones(4, 24, DIM * 2),  # targets
        (4, 24, DIM * 2),  # expected logits shape
    ),
]


@pytest.mark.parametrize("module,inp,outp,exp_logits_shape,exp_prob_shape", TEST_CASES_CLS)
def test_ffn_cls_head_smoke(module, inp, outp, exp_logits_shape, exp_prob_shape):
    assert module.num_decoder_layers == inp.shape[0]
    assert module.aux_mlp is None

    if module.encoder_mlp is not None:
        assert module.encoder_mlp is module.mlp

    # forward entire output sequence in one go (DETR)
    pred_logits = module(inp)
    assert pred_logits.shape == exp_logits_shape

    # compute loss for one head
    # logits [B, R, C]; targets [B, R]
    loss = module.compute_loss(pred_logits[-1], outp)
    sum(loss.values()).backward()

    # last head output will be passsed to postprocess logits
    pred_probs = module.postprocess_logits(pred_logits[-1])
    assert pred_probs.shape == exp_prob_shape


@pytest.mark.parametrize("module,inp,outp,exp_logits_shape,exp_prob_shape,exp_aux_shape", TEST_CASES_CLS_SEP)
def test_ffn_cls_head_separate_smoke(module, inp, outp, exp_logits_shape, exp_prob_shape, exp_aux_shape):
    assert module.num_decoder_layers == inp.shape[0]
    assert len(module.aux_mlp) == NUM_DECODER_LAYERS - 1
    for aux_module in module.aux_mlp:
        assert aux_module is not module.mlp

    if module.encoder_mlp is not None:
        assert module.encoder_mlp is not module.aux_mlp

    # forward each output sequence separately (Deformable DETR)
    pred_logits_aux = module(inp[0], layer=0)
    assert pred_logits_aux.shape == exp_aux_shape

    pred_logits = module(inp[1], layer=1)
    assert pred_logits.shape == exp_logits_shape

    # compute loss for one head
    # logits [B, R, C]; targets [B, R]
    loss = module.compute_loss(pred_logits, outp)
    sum(loss.values()).backward()

    # last head output will be passsed to postprocess logits
    pred_probs = module.postprocess_logits(pred_logits)
    assert pred_probs.shape == exp_prob_shape


@pytest.mark.parametrize("module,inp,outp,exp_logits_shape", TEST_CASES_REG)
def test_ffn_reg_head_smoke(module, inp, outp, exp_logits_shape):
    assert module.num_decoder_layers == inp.shape[0]
    assert module.aux_mlp is None

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


@pytest.mark.parametrize("module,inp,outp,exp_logits_shape", TEST_CASES_REG_SEP)
def test_ffn_reg_head_seperate_smoke(module, inp, outp, exp_logits_shape):
    assert module.num_decoder_layers == inp.shape[0]
    assert len(module.aux_mlp) == NUM_DECODER_LAYERS - 1
    for aux_module in module.aux_mlp:
        assert aux_module is not module.mlp

    # forward each output sequence separately (Deformable DETR)
    pred_logits_aux = module(inp[0], layer=0)
    assert pred_logits_aux.shape == exp_logits_shape

    pred_logits = module(inp[1], layer=1)
    assert pred_logits.shape == exp_logits_shape

    # compute loss for one head
    # logits [B, R, C]; targets [B, R]
    loss = module.compute_loss(
        preds=pred_logits,
        targets=outp,
        pred_boxes=pred_logits,
        target_boxes=outp,
    )
    sum(loss.values()).backward()
