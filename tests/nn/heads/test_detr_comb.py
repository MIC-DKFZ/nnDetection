import pytest
import torch

import nndet.core.ops_torch as ops_torch
from nndet.core.boxes.criterions.box import L1RegCriterion
from nndet.core.boxes.criterions.cls import SimpleClassCriterionSigmoid
from nndet.core.boxes.matcher1to1.hungarian import HungarianMatcher
from nndet.core.post.detr import MaxFGBoxPost
from nndet.nn.heads.classifier.ffn import BCEFFNClassifier
from nndet.nn.heads.detr.base import DETRHead
from nndet.nn.heads.detr.cdetr import ConditionalDETRHead
from nndet.nn.heads.regressor.ffn import L1FFNRegressor
from nndet.nn.layers.linear import LayerLinearReluDrop
from nndet.utils.enums import AuxLossNorm

IN_CHANNELS = 8
INTERNAL_CHANNELS = 16
NUM_CLASSES = 2
DIM = 3


def prepare_modules():
    classifier = BCEFFNClassifier(
        linear=LayerLinearReluDrop,
        in_channels=IN_CHANNELS,
        internal_channels=INTERNAL_CHANNELS,
        num_classes=NUM_CLASSES,
    )
    regressor = L1FFNRegressor(
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
def detr_head():
    classifier, regressor, matcher, box_post = prepare_modules()
    return DETRHead(
        classifier=classifier,
        regressor=regressor,
        matcher=matcher,
        box_post=box_post,
        aux_loss=True,
        norm_cls_loss_by_num_boxes=True,
        norm_reg_loss_by_num_boxes=True,
    )


@pytest.fixture
def cdetr_head():
    classifier, regressor, matcher, box_post = prepare_modules()
    return ConditionalDETRHead(
        classifier=classifier,
        regressor=regressor,
        matcher=matcher,
        box_post=box_post,
        aux_loss=True,
        norm_cls_loss_by_num_boxes=True,
        norm_reg_loss_by_num_boxes=True,
    )


@pytest.fixture
def sig75() -> float:
    val = torch.tensor(0.75)
    sig_inv = ops_torch.inverse_sigmoid(val)
    assert torch.allclose(torch.functional.F.sigmoid(sig_inv), val)
    return sig_inv.item()


def test_prepare_targets(detr_head):
    target_boxes_point = [
        torch.tensor(
            [
                [0, 0, 1, 1, 0, 1],
                [1, 1, 2, 2, 1, 2],
            ],
            dtype=torch.float,
        ),
        torch.tensor([[0, 0, 1, 1, 0, 1]]),
    ]
    target_labels_num = [
        torch.tensor([0, 1]),
        torch.tensor([1]),
    ]
    img_shape = (2, 2, 2)

    prepared_boxes_center, prepared_labels = detr_head.prepare_targets(
        target_boxes=target_boxes_point,
        target_labels=target_labels_num,
        img_shape=img_shape,
    )

    assert len(prepared_boxes_center) == 2
    assert len(prepared_labels) == 2

    assert prepared_boxes_center[0].shape == (2, 6)
    assert prepared_boxes_center[1].shape == (1, 6)

    assert prepared_labels[0].shape == (2,)
    assert prepared_labels[1].shape == (1,)

    # scaled + conversion
    assert torch.allclose(
        prepared_boxes_center[0],
        torch.tensor(
            [
                [
                    0.25,
                    0.25,
                    0.5,
                    0.5,
                    0.25,
                    0.5,
                ],  # point: [0, 0, 0.5, 0.5, 0, 0.5],
                [
                    0.75,
                    0.75,
                    0.5,
                    0.5,
                    0.75,
                    0.5,
                ],  # point: [0.5, 0.5, 1, 1, 0.5, 1],
            ]
        ),
    )
    assert torch.allclose(
        prepared_boxes_center[1],
        torch.tensor(
            [
                [
                    0.25,
                    0.25,
                    0.5,
                    0.5,
                    0.25,
                    0.5,
                ],  # point: [0, 0, 0.5, 0.5, 0, 0.5],
            ]
        ),
    )
    # added background
    assert torch.allclose(prepared_labels[0], torch.tensor([1, 2]))
    assert torch.allclose(prepared_labels[1], torch.tensor([2]))


def test_get_src_permutation_idx(detr_head):
    example_indices = [
        (torch.tensor([0, 1]), torch.tensor([0, 1])),
        (None, None),
        (torch.tensor([0, 1]), torch.tensor([1, 0])),
        (None, None),
        (torch.tensor([1, 0]), torch.tensor([0, 1])),
    ]
    batch_indices, predictions_indices = detr_head._get_src_permutation_idx(
        indices=example_indices,
    )
    expected_batch_indices = torch.tensor([0, 0, 2, 2, 4, 4])
    expected_predictions_indices = torch.tensor([0, 1, 0, 1, 1, 0])

    assert torch.allclose(batch_indices, expected_batch_indices)
    assert torch.allclose(predictions_indices, expected_predictions_indices)


def test_get_src_permutation_idx_empty(detr_head):
    example_indices = [(None, None), (None, None), (None, None)]
    batch_indices, predictions_indices = detr_head._get_src_permutation_idx(
        indices=example_indices,
    )

    assert batch_indices.numel() == 0
    assert predictions_indices.numel() == 0


def test_compute_class_loss(detr_head):
    example_labels = [  # 0 = background
        torch.tensor([1, 2]),
        torch.tensor([]),
        torch.tensor([1, 2]),
    ]  # List[X]
    example_indices = [
        (torch.tensor([0, 2]), torch.tensor([0, 1])),
        (None, None),
        (torch.tensor([1, 2]), torch.tensor([1, 0])),
    ]
    exampled_logits = torch.zeros(3, 3, 2)  # B, R, NC
    exampled_logits[0, 0, 0] = 1
    exampled_logits[0, 2, 1] = 1
    exampled_logits[2, 1, 1] = 1
    exampled_logits[2, 2, 0] = 1
    num_boxes = 4

    loss = detr_head.compute_class_loss(
        pred_logits=exampled_logits,
        target_labels=example_labels,
        indices=example_indices,
        num_boxes_all=num_boxes,
    )["ffn_bce"]

    expected_loss = (
        torch.functional.F.binary_cross_entropy_with_logits(
            torch.cat([torch.zeros(14), torch.ones(4)]),  # inp
            torch.cat([torch.zeros(14), torch.ones(4)]),  # target
            reduction="sum",
        )
        / num_boxes
    )

    assert torch.allclose(loss, expected_loss)


def test_compute_box_loss(detr_head):
    example_targets = [
        torch.tensor(
            [
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [1.5, 1.5, 0.5, 0.5, 1.5, 0.5],
            ]
        ),
        torch.tensor([]),
        torch.tensor(
            [
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [1.5, 1.5, 0.5, 0.5, 1.5, 0.5],
            ]
        ),
    ]
    example_indices = [
        (torch.tensor([0, 2]), torch.tensor([0, 1])),
        (None, None),
        (torch.tensor([1, 2]), torch.tensor([1, 0])),
    ]
    example_boxes = torch.zeros(3, 3, DIM * 2)
    example_boxes[0, 0] = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    example_boxes[0, 2] = torch.tensor([1.5, 1.5, 0.5, 0.5, 1.5, 0.5])
    example_boxes[2, 1] = torch.tensor([1.5, 1.5, 0.5, 0.5, 1.5, 0.5])
    example_boxes[2, 2] = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    num_boxes = 4

    loss = detr_head.compute_box_loss(
        pred_coords=example_boxes,
        target_boxes=example_targets,
        indices=example_indices,
        num_boxes_all=num_boxes,
    )["ffn_reg_l1"]

    assert torch.allclose(loss, torch.tensor(0, dtype=torch.float))


class TestForward:
    num_dec = 4
    num_det = 5
    bs = 3

    def test_detr_forward(self, detr_head):
        self._forward_assert_module(detr_head)

    def test_cdetr_forward(self, cdetr_head):
        refs_ccddcd_norm = torch.rand(self.bs, self.num_det, DIM)
        self._forward_assert_module(cdetr_head, refs_ccddcd_norm=refs_ccddcd_norm)

    def _forward_assert_module(self, module, refs_ccddcd_norm=None):

        example_sequence = torch.rand(self.num_dec, self.bs, self.num_det, IN_CHANNELS)
        expected_logit_shape = (self.bs, self.num_det, NUM_CLASSES)
        expected_coords_shape = (self.bs, self.num_det, DIM * 2)

        output = module(example_sequence, refs_ccddcd_norm=refs_ccddcd_norm)

        assert output["pred_cls_logits"].shape == expected_logit_shape
        assert output["pred_box_coords"].shape == expected_coords_shape
        assert len(output["aux_outputs"]) == self.num_dec - 1
        for i in range(self.num_dec - 1):
            assert output["aux_outputs"][i]["pred_cls_logits"].shape == expected_logit_shape
            assert output["aux_outputs"][i]["pred_box_coords"].shape == expected_coords_shape


def test_postprocess_for_inference(detr_head, sig75):
    num_det = 2
    bs = 2

    example_boxes = torch.zeros(bs, num_det, DIM * 2, dtype=torch.float)  # [B, R, num_classes]
    example_boxes[0, 0] = torch.tensor([0.25, 0.25, 0.5, 0.5, 0.25, 0.5])  # point: [0, 0, 0.5, 0.5, 0, 0.5] pre scale
    example_boxes[0, 1] = torch.tensor([0.75, 0.75, 0.5, 0.5, 0.75, 0.5])  # point: [0.5, 0.5, 1, 1, 0.5, 1] pre scale
    example_boxes[1, 0] = torch.tensor([0.75, 0.75, 0.5, 0.5, 0.75, 0.5])  # point: [0.5, 0.5, 1, 1, 0.5, 1] pre scale
    example_boxes[1, 1] = torch.tensor([0.25, 0.25, 0.5, 0.5, 0.25, 0.5])  # point: [0, 0, 0.5, 0.5, 0, 0.5] pre scale

    example_logits = torch.zeros(bs, num_det, NUM_CLASSES, dtype=torch.float)  # [B, R, dims * 2]
    example_logits[0, 0, 0] = sig75
    example_logits[0, 1, 1] = sig75
    example_logits[1, 0, 0] = sig75
    example_logits[1, 1, 1] = sig75

    expected_probs = [torch.tensor([0.75, 0.75]), torch.tensor([0.75, 0.75])]
    expected_labels = [torch.tensor([0, 1]), torch.tensor([0, 1])]
    expected_boxes = [
        torch.tensor(
            [
                [0, 0, 1, 1, 0, 1],
                [1, 1, 2, 2, 1, 2],
            ],
            dtype=torch.float,
        ),
        torch.tensor(
            [
                [1, 1, 2, 2, 1, 2],
                [0, 0, 1, 1, 0, 1],
            ],
            dtype=torch.float,
        ),
    ]

    predictions = detr_head.postprocess_for_inference(
        pred_detection={
            "pred_cls_logits": example_logits,
            "pred_box_coords": example_boxes,
        },
        img_shape=(2, 2, 2),
    )

    assert len(predictions["pred_boxes"]) == 2
    assert len(predictions["pred_scores"]) == 2
    assert len(predictions["pred_labels"]) == 2

    for i in range(2):
        assert torch.allclose(predictions["pred_boxes"][i], expected_boxes[i])
        assert torch.allclose(predictions["pred_scores"][i], expected_probs[i])
        assert torch.allclose(predictions["pred_labels"][i], expected_labels[i])


def test_format_scale_aux_losses_none(detr_head):
    detr_head.scale_aux_loss = AuxLossNorm.NONE

    losses = {"cls": torch.tensor(1.0), "reg": torch.tensor(2.0)}
    losses_aux = detr_head.format_scale_aux_losses(losses, num_aux_outputs=4, aux_idx=2)

    assert len(losses_aux) == 2
    assert torch.allclose(losses_aux["aux_cls_2"], losses["cls"])
    assert torch.allclose(losses_aux["aux_reg_2"], losses["reg"])


def test_format_scale_aux_losses_mean(detr_head):
    detr_head.scale_aux_loss = AuxLossNorm.MEAN

    losses = {"cls": torch.tensor(1.0), "reg": torch.tensor(2.0)}
    losses_aux = detr_head.format_scale_aux_losses(losses, num_aux_outputs=4, aux_idx=2)

    assert len(losses_aux) == 2
    assert torch.allclose(losses_aux["aux_cls_2"], losses["cls"] / 4)
    assert torch.allclose(losses_aux["aux_reg_2"], losses["reg"] / 4)


def test_format_scale_aux_losses_reduced(detr_head):
    detr_head.scale_aux_loss = AuxLossNorm.REDUCED

    losses = {"cls": torch.tensor(1.0), "reg": torch.tensor(2.0)}
    losses_aux = detr_head.format_scale_aux_losses(losses, num_aux_outputs=4, aux_idx=2)

    assert len(losses_aux) == 2
    assert torch.allclose(losses_aux["aux_cls_2"], losses["cls"] / 3)
    assert torch.allclose(losses_aux["aux_reg_2"], losses["reg"] / 3)
