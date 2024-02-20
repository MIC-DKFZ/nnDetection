from typing import Optional

import pytest
import torch

from nndet.core.boxes.criterions.box import L1RegCriterion
from nndet.core.boxes.criterions.cls import SimpleClassCriterionSigmoid
from nndet.core.boxes.matcher1to1.hungarian import HungarianMatcher
from nndet.core.post.detr import MaxFGBoxPost
from nndet.nn.heads.classifier.ffn import BCEFFNClassifier
from nndet.nn.heads.detr.deformable_detr import DeformableDETRHead
from nndet.nn.heads.regressor.ffn import L1FFNRegressor
from nndet.nn.layers.linear import LayerLinearReluDrop

IN_CHANNELS = 8
INTERNAL_CHANNELS = 16
NUM_CLASSES = 2
DIM = 3
N_DET = 12
N_DECODER = 3


class OnesClassifier(BCEFFNClassifier):
    def forward(self, features: torch.Tensor, layer: Optional[int] = None) -> torch.Tensor:
        bs, num_p, _ = features.shape
        return torch.ones((bs, num_p, self.num_classes), device=features.device, dtype=features.dtype)


class AddOneModule(torch.nn.Module):
    def forward(self, x):
        return x + 1


class SeqRegressor(L1FFNRegressor):
    _box_norm_fn = AddOneModule()
    _inverse_box_norm_fn = AddOneModule()

    def forward(self, features: torch.Tensor, layer: Optional[int] = None):
        bs, num_p, _ = features.shape
        return torch.zeros((bs, num_p, 2 * self.dim), device=features.device, dtype=features.dtype).fill_(0.5)


def prepare_modules():
    classifier = OnesClassifier(
        linear=LayerLinearReluDrop,
        in_channels=IN_CHANNELS,
        internal_channels=INTERNAL_CHANNELS,
        num_classes=NUM_CLASSES,
    )
    regressor = SeqRegressor(
        linear=LayerLinearReluDrop,
        in_channels=IN_CHANNELS,
        internal_channels=INTERNAL_CHANNELS,
        dim=DIM,
    )
    matcher = HungarianMatcher(
        class_criterion=[SimpleClassCriterionSigmoid(1.0)],
        box_criterion=[L1RegCriterion(1.0)],
    )
    box_post = MaxFGBoxPost()
    return classifier, regressor, matcher, box_post


@pytest.fixture
def deformable_detr_head():
    classifier, regressor, matcher, box_post = prepare_modules()
    return DeformableDETRHead(
        classifier=classifier,
        regressor=regressor,
        matcher=matcher,
        box_post=box_post,
        aux_loss=True,
        norm_cls_loss_by_num_boxes=True,
        norm_reg_loss_by_num_boxes=True,
    )


def test_deformable_detr_head(deformable_detr_head):
    bs = 2
    out_sequence = torch.rand(N_DECODER, bs, N_DET, IN_CHANNELS)
    refs_ccddcd_norm = torch.zeros(N_DECODER + 1, bs, N_DET, DIM * 2)

    pred_detection = deformable_detr_head(out_sequence=out_sequence, refs_ccddcd_norm=refs_ccddcd_norm)

    expected_cls_logits = torch.ones((bs, N_DET, NUM_CLASSES))
    expected_box_coords = torch.zeros((bs, N_DET, DIM * 2)).fill_(2.5)
    assert torch.allclose(pred_detection["pred_cls_logits"], expected_cls_logits)
    assert torch.allclose(pred_detection["pred_box_coords"], expected_box_coords)

    assert len(pred_detection["aux_outputs"]) == N_DECODER - 1
    for aux in pred_detection["aux_outputs"]:
        assert torch.allclose(aux["pred_cls_logits"], expected_cls_logits)
        assert torch.allclose(aux["pred_box_coords"], expected_box_coords)
