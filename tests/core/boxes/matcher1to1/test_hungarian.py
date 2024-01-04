import pytest
import torch

from nndet.core.boxes.criterions.box import L1RegCriterion
from nndet.core.boxes.criterions.cls import SimpleClassCriterionSigmoid
from nndet.core.boxes.matcher1to1.hungarian import HungarianMatcher


@pytest.fixture
def class_criterion():
    return [SimpleClassCriterionSigmoid(loss_weight=1.0)]


@pytest.fixture
def box_criterion():
    return [L1RegCriterion(loss_weight=1.0)]


def test_hungarian_matcher_cls(class_criterion, box_criterion):
    matcher = HungarianMatcher(class_criterion, box_criterion)

    pred_logits = torch.tensor(
        [
            [
                [1.0],
                [0.0],
            ],
            [
                [0.0],
                [1.0],
            ],
        ],
        dtype=torch.float,
    )  # B=2, R=2, C=1
    pred_coords = torch.zeros((2, 2, 6))
    target_labels = [
        torch.tensor([1], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
    ]
    target_boxes = [
        torch.tensor([[0, 0, 1, 1, 0, 1]], dtype=torch.float),
        torch.tensor([[0, 0, 1, 1, 0, 1]], dtype=torch.float),
    ]

    indices, _ = matcher.match(pred_logits, pred_coords, target_boxes, target_labels)

    assert len(indices) == 2
    assert tuple(indices[0][0].shape) == (1,)
    assert tuple(indices[0][1].shape) == (1,)
    assert torch.allclose(indices[0][0], torch.tensor([0], dtype=torch.long))
    assert torch.allclose(indices[0][1], torch.tensor([0], dtype=torch.long))
    assert tuple(indices[1][0].shape) == (1,)
    assert tuple(indices[1][1].shape) == (1,)
    assert torch.allclose(indices[1][0], torch.tensor([1], dtype=torch.long))
    assert torch.allclose(indices[1][1], torch.tensor([0], dtype=torch.long))


def test_hungarian_matcher_reg(class_criterion, box_criterion):
    matcher = HungarianMatcher(class_criterion, box_criterion)

    pred_logits = torch.tensor(
        [
            [
                [1.0],
                [1.0],
            ],
            [
                [1.0],
                [1.0],
            ],
        ],
        dtype=torch.float,
    )  # B=2, R=2, C=1
    pred_coords = torch.tensor(
        [
            [[7, 7, 10, 10, 7, 10], [0, 0, 1, 1, 0, 1]],
            [[0, 0, 1, 1, 0, 1], [7, 7, 10, 10, 7, 10]],
        ],
        dtype=torch.float,
    )  # B=2, R=2, dims=2*3
    target_labels = [
        torch.tensor([1], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
    ]
    target_boxes = [
        torch.tensor([[0, 0, 1, 1, 0, 1]], dtype=torch.float),
        torch.tensor([[0, 0, 1, 1, 0, 1]], dtype=torch.float),
    ]

    indices, _ = matcher.match(pred_logits, pred_coords, target_boxes, target_labels)

    assert len(indices) == 2
    assert tuple(indices[0][0].shape) == (1,)
    assert tuple(indices[0][1].shape) == (1,)
    assert torch.allclose(indices[0][0], torch.tensor([1], dtype=torch.long))
    assert torch.allclose(indices[0][1], torch.tensor([0], dtype=torch.long))
    assert tuple(indices[1][0].shape) == (1,)
    assert tuple(indices[1][1].shape) == (1,)
    assert torch.allclose(indices[1][0], torch.tensor([0], dtype=torch.long))
    assert torch.allclose(indices[1][1], torch.tensor([0], dtype=torch.long))


def test_hungarian_matcher_reg_mul_gt(class_criterion, box_criterion):
    matcher = HungarianMatcher(class_criterion, box_criterion)

    pred_logits = torch.tensor(
        [
            [[1.0], [1.0], [1.0]],
            [[1.0], [1.0], [1.0]],
        ],
        dtype=torch.float,
    )  # B=2, R=3, C=1
    pred_coords = torch.tensor(
        [
            [[7, 7, 10, 10, 7, 10], [0, 0, 1, 1, 0, 1], [7, 7, 10, 10, 7, 10]],
            [[0, 0, 1, 1, 0, 1], [7, 7, 10, 10, 7, 10], [12, 12, 15, 15, 12, 15]],
        ],
        dtype=torch.float,
    )  # B=2, R=2, dims=2*3
    target_labels = [
        torch.tensor([1], dtype=torch.long),
        torch.tensor([1, 1], dtype=torch.long),
    ]
    target_boxes = [
        torch.tensor([[0, 0, 1, 1, 0, 1]], dtype=torch.float),
        torch.tensor([[0, 0, 1, 1, 0, 1], [12, 12, 15, 15, 12, 15]], dtype=torch.float),
    ]

    indices, _ = matcher.match(pred_logits, pred_coords, target_boxes, target_labels)

    assert len(indices) == 2
    assert tuple(indices[0][0].shape) == (1,)
    assert tuple(indices[0][1].shape) == (1,)
    assert torch.allclose(indices[0][0], torch.tensor([1], dtype=torch.long))
    assert torch.allclose(indices[0][1], torch.tensor([0], dtype=torch.long))
    assert tuple(indices[1][0].shape) == (2,)
    assert tuple(indices[1][1].shape) == (2,)
    assert torch.allclose(indices[1][0], torch.tensor([0, 2], dtype=torch.long))
    assert torch.allclose(indices[1][1], torch.tensor([0, 1], dtype=torch.long))
