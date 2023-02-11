import pytest
import torch

from nndet.nn.heads.masker.roi import (
    BCEAgnosticMasker,
    BCESpecificMasker,
    BDiCEAgnosticMasker,
    BDiCESpecificMasker,
    Masker,
)
from nndet.nn.layers.conv.instance import ConvInstanceLReLU
from nndet.nn.layers.wrapper import Generator

NUM_CLASSES = 4
ROI_SHAPE = (12, 16, 8, 8, 8)
TARGET_SHAPE = (12, 16, 16, 16)
EXAMPLE_CONFIG = {
    "num_classes": NUM_CLASSES,
    "conv": Generator(ConvInstanceLReLU, 3),
    "in_channels": 16,
    "internal_channels": 32,
    "num_convs": 1,
    "add_norm": True,
}

EXPECTED_SHAPE_AGNOSTIC = (12, 1, 16, 16, 16)
EXPECTED_SHAPE_SPECIFIC = (12, NUM_CLASSES, 16, 16, 16)
EXPECTED_SHAPE_PROBS = (12, 16, 16, 16)

TEST_CASES = [
    BCEAgnosticMasker(**EXAMPLE_CONFIG, prior_prob=None),
    BCEAgnosticMasker(**EXAMPLE_CONFIG, prior_prob=0.01),
    BCESpecificMasker(**EXAMPLE_CONFIG, prior_prob=None),
    BCESpecificMasker(**EXAMPLE_CONFIG, prior_prob=0.01),
    BDiCEAgnosticMasker(**EXAMPLE_CONFIG),
    BDiCESpecificMasker(**EXAMPLE_CONFIG),
]


@pytest.mark.parametrize("module", TEST_CASES)
def test_forward_backward_smoke(module: Masker):
    # check repr
    str(module)

    roi_batch = torch.zeros(ROI_SHAPE)
    target_roi = torch.zeros(TARGET_SHAPE)
    labels_roi = torch.tensor([i % NUM_CLASSES for i in range(ROI_SHAPE[0])])

    preds = module(roi_batch)[0]  # N, C, dims
    if module.is_class_agnostic():
        assert module.get_output_channels() == 1
        assert tuple(preds.shape) == EXPECTED_SHAPE_AGNOSTIC
    else:
        assert module.get_output_channels() == NUM_CLASSES
        assert tuple(preds.shape) == EXPECTED_SHAPE_SPECIFIC

    loss = module.compute_loss(
        pred_logits=preds,
        target_masks=target_roi,
        target_labels=labels_roi,
    )
    sum(loss.values()).backward()


@pytest.mark.parametrize("module", TEST_CASES)
def test_logits_to_probs_smoke(module: Masker):
    roi_batch = torch.zeros(ROI_SHAPE)
    labels_roi = torch.tensor([i % NUM_CLASSES for i in range(ROI_SHAPE[0])])

    preds = module(roi_batch)[0]  # N, C, dims
    probs = module.logits_to_probs(logits=preds, labels=labels_roi)
    assert tuple(probs.shape) == EXPECTED_SHAPE_PROBS
