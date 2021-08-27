import pytest

import torch

from nndet.core.boxes.matcher import ATSSMatcher


@pytest.fixture
def anchors():
    anchors = torch.tensor(
        [
            [0, 0, 4, 4, 0, 4],
            [4, 4, 8, 8, 0, 4],
            [0, 4, 4, 8, 0, 4],
            [4, 0, 8, 4, 0, 4],
            [0, 0, 8, 8, 0, 4],
        ]
    )
    return anchors


@pytest.mark.parametrize("matcher", [ATSSMatcher(4)])
def test_no_gt(matcher, anchors):
    match_quality_matrix, matches = matcher(
        torch.tensor([]), anchors, num_anchors_per_level=[4, 1], num_anchors_per_loc=1
    )
    assert match_quality_matrix.numel() == 0
    expected = torch.empty(anchors.shape[0], dtype=torch.int64).fill_(
        matcher.BELOW_LOW_THRESHOLD
    )
    assert matches.allclose(expected)


@pytest.mark.parametrize("matcher", [ATSSMatcher(4)])
def test_matching(matcher, anchors):
    match_quality_matrix, matches = matcher(
        anchors[[0, 1, 2]],
        anchors,
        num_anchors_per_level=[4, 1],
        num_anchors_per_loc=1,
    )

    iou_l = (4 * 4 * 4) / (8 * 8 * 4)
    expected_ious = torch.tensor(
        [[1, 0, 0, 0, iou_l], [0, 1, 0, 0, iou_l], [0, 0, 1, 0, iou_l]]
    )
    assert match_quality_matrix.allclose(expected_ious)
    expected = torch.empty(anchors.shape[0], dtype=torch.int64).fill_(
        matcher.BELOW_LOW_THRESHOLD
    )
    expected[0] = 0
    expected[1] = 1
    expected[2] = 2
    assert matches.allclose(expected)
