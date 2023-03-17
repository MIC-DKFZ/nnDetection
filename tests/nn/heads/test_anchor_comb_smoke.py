from typing import Type

import pytest
import torch

from nndet.core.boxes.coder import BoxCoderND
from nndet.core.boxes.sampler import HardNegativeSamplerBatched
from nndet.nn.heads.classifier.dense import BCECLassifier
from nndet.nn.heads.comb.anchor_all import BoxHeadAll
from nndet.nn.heads.comb.anchor_sampled import BoxHeadHNM, BoxHeadHNMV2
from nndet.nn.heads.comb.base import AnchorHead
from nndet.nn.heads.regressor.dense import L1Regressor
from nndet.nn.layers.conv import ConvInstanceLReLU
from nndet.nn.layers.wrapper import Generator

NUM_CLASSES = 2
DIM = 3
N = 4
INPUT_SIZE_TENSORS = [(N, 8, 4, 4, 4), (N, 8, 2, 2, 2)]
TARGET_SIZE_CLS = N
TARGET_SIZE_REG = (N, DIM * 2)

ANCHORS_PER_POS = 3
ANCHORS_IMG = ANCHORS_PER_POS * 4 * 4 * 4 + ANCHORS_PER_POS * 2 * 2 * 2
ANCHORS = [torch.tensor([[1.0, 1.0, 2.0, 2.0, 1.0, 2.0]]).expand(ANCHORS_IMG, DIM * 2) for _ in range(N)]

EXAMPLE_CONFIG = {
    "conv": Generator(ConvInstanceLReLU, 3),
    "in_channels": 8,
    "internal_channels": 16,
    "num_classes": NUM_CLASSES,
    "num_levels": 2,
    "anchors_per_pos": ANCHORS_PER_POS,
}


@pytest.fixture
def classifier():
    return BCECLassifier(
        **EXAMPLE_CONFIG,
        add_norm=True,
    )


@pytest.fixture
def regressor():
    return L1Regressor(
        **EXAMPLE_CONFIG,
        add_norm=True,
    )


@pytest.fixture
def sampler():
    return HardNegativeSamplerBatched(
        batch_size_per_image=32,
        positive_fraction=0.5,
    )


@pytest.fixture
def coder():
    return BoxCoderND([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


@pytest.mark.parametrize("module_cls", [BoxHeadAll])
def test_head_all(module_cls: Type[AnchorHead], classifier, regressor, coder):
    head: AnchorHead = module_cls(
        classifier=classifier,
        regressor=regressor,
        coder=coder,
    )
    fmaps = [torch.zeros(i, dtype=torch.float) for i in INPUT_SIZE_TENSORS]
    preds = head(fmaps)

    assert "box_deltas" in preds
    assert "box_logits" in preds
    assert tuple(preds["box_deltas"].shape) == (N * ANCHORS_IMG, DIM * 2)
    assert tuple(preds["box_logits"].shape) == (N * ANCHORS_IMG, NUM_CLASSES)

    preds_post = head.postprocess_for_inference(preds, ANCHORS)
    assert "pred_boxes" in preds_post
    assert "pred_probs" in preds_post
    assert tuple(preds["box_deltas"].shape) == (N * ANCHORS_IMG, DIM * 2)
    assert tuple(preds["box_logits"].shape) == (N * ANCHORS_IMG, NUM_CLASSES)

    matched_gt_labels = [torch.tensor([x % (NUM_CLASSES + 1) for x in range(ANCHORS_IMG)]) for _ in range(N)]
    loss, pos_inds, neg_inds = head.compute_loss(
        prediction=preds,
        matched_gt_labels=matched_gt_labels,
        matched_gt_boxes=ANCHORS,
        anchors=ANCHORS,
    )
    assert "reg" in loss
    assert "cls" in loss


@pytest.mark.parametrize("module_cls", [BoxHeadHNM, BoxHeadHNMV2])
def test_head_hnm(module_cls: Type[AnchorHead], classifier, regressor, coder, sampler):
    head: AnchorHead = module_cls(
        classifier=classifier,
        regressor=regressor,
        coder=coder,
        sampler=sampler,
    )
    fmaps = [torch.zeros(i, dtype=torch.float) for i in INPUT_SIZE_TENSORS]
    preds = head(fmaps)

    assert "box_deltas" in preds
    assert "box_logits" in preds
    assert tuple(preds["box_deltas"].shape) == (N * ANCHORS_IMG, DIM * 2)
    assert tuple(preds["box_logits"].shape) == (N * ANCHORS_IMG, NUM_CLASSES)

    preds_post = head.postprocess_for_inference(preds, ANCHORS)
    assert "pred_boxes" in preds_post
    assert "pred_probs" in preds_post
    assert tuple(preds["box_deltas"].shape) == (N * ANCHORS_IMG, DIM * 2)
    assert tuple(preds["box_logits"].shape) == (N * ANCHORS_IMG, NUM_CLASSES)

    matched_gt_labels = [torch.tensor([x % (NUM_CLASSES + 1) for x in range(ANCHORS_IMG)]) for _ in range(N)]
    loss, pos_inds, neg_inds = head.compute_loss(
        prediction=preds,
        matched_gt_labels=matched_gt_labels,
        matched_gt_boxes=ANCHORS,
        anchors=ANCHORS,
    )
    assert "reg" in loss
    assert "cls" in loss
