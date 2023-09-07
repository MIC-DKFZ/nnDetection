import numpy as np
import pytest
from pytest_mock import MockerFixture

import nndet.core.ops_np as ops_np
from nndet.eval.matching import EvalMatchingPerElementGreedyScoreNP


@pytest.fixture
def example():
    pred_boxes = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.5, 0.5, 1.1, 1.1],
            [0.9, 0.8, 2.0, 2.0],
        ]
    )
    pred_scores = np.array([0.5, 0.9, 0.3])
    gt_boxes = np.array(
        [
            [0.5, 0, 1.5, 1],
            [0.5, 0.5, 1.0, 1.0],
            [1.0, 1.0, 2.0, 2.0],
        ]
    )
    return pred_boxes, pred_scores, gt_boxes


class TestMatching:
    def test_box_matching_batch(self, mocker: MockerFixture):
        matching = EvalMatchingPerElementGreedyScoreNP(iou_fn=ops_np.box_iou_np)
        matching._matching_single_image_single_class = mocker.Mock(return_value=0)
        # iou_fn = mocker.Mock(return_value=None)
        iou_tresholds = [0.1, 0.5]

        pred_boxes = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
        pred_classes = np.array([1, 0])
        pred_scores = np.array([1.0, 1.0])

        gt_boxes = np.array([[4, 5, 6, 7], [4, 5, 6, 7]])
        gt_classes = np.array([0, 1])
        gt_ignore = np.array([0, 0])

        res = matching.match(
            # iou_fn,
            iou_tresholds,
            [pred_boxes],
            [pred_classes],
            [pred_scores],
            [gt_boxes],
            [gt_classes],
            [gt_ignore],
            ["case0"],
        )

        assert len(res) == 1
        assert {"case_id": "case0", 0: 0, 1: 0} == res[0]
        matching._matching_single_image_single_class.assert_called()

    def test_box_matching_batch_no_gt(self, mocker: MockerFixture):
        matching = EvalMatchingPerElementGreedyScoreNP(iou_fn=ops_np.box_iou_np)
        matching._matching_no_gt = mocker.Mock(return_value=0)
        iou_tresholds = [0.1, 0.5]

        pred_boxes = np.array([[0.0, 1.0, 2.0, 3.0]])
        pred_classes = np.array([0])
        pred_scores = np.array([1.0])

        gt_boxes = np.array([[]]).reshape(-1, 6)
        gt_classes = np.array([])
        gt_ignore = np.array([])

        res = matching.match(
            iou_tresholds,
            [pred_boxes],
            [pred_classes],
            [pred_scores],
            [gt_boxes],
            [gt_classes],
            [gt_ignore],
        )

        assert len(res) == 1
        assert res[0] == {"case_id": None, 0: 0}
        matching._matching_no_gt.assert_called()

    def test_box_matching_batch_no_pred(self, mocker: MockerFixture):
        matching = EvalMatchingPerElementGreedyScoreNP(iou_fn=ops_np.box_iou_np)
        matching._matching_no_pred = mocker.Mock(return_value=0)
        iou_tresholds = [0.1, 0.5]

        pred_boxes = np.array([[]]).reshape(-1, 6)
        pred_classes = np.array([])
        pred_scores = np.array([])

        gt_boxes = np.array([[0.0, 1.0, 2.0, 3.0]])
        gt_classes = np.array([0])
        gt_ignore = np.array([1.0])

        res = matching.match(
            iou_tresholds,
            [pred_boxes],
            [pred_classes],
            [pred_scores],
            [gt_boxes],
            [gt_classes],
            [gt_ignore],
        )

        assert len(res) == 1
        assert res[0] == {"case_id": None, 0: 0}
        matching._matching_no_pred.assert_called()

    def test_box_matching_single_image_single_class_no_ignore(self, example):
        matching = EvalMatchingPerElementGreedyScoreNP(iou_fn=ops_np.box_iou_np)

        pred_boxes, pred_scores, gt_boxes = example
        gt_ignore = np.array([0, 0, 0])
        iou_thresholds = [0.1, 0.5]

        res = matching._matching_single_image_single_class(
            iou_thresholds=iou_thresholds,
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            gt_boxes=gt_boxes,
            gt_ignore=gt_ignore,
            case_id="example",
        )

        assert np.isclose(res["dtMatches"], [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]]).all()
        assert np.isclose(res["gtMatches"], [[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]).all()
        assert np.isclose(res["dtScores"], [0.9, 0.5, 0.3]).all()
        assert np.isclose(res["gtIgnore"], [0.0, 0.0, 0.0]).all()
        assert np.isclose(res["dtIgnore"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).all()

    def test_box_matching_single_image_single_class_ignore(self, example):
        matching = EvalMatchingPerElementGreedyScoreNP(iou_fn=ops_np.box_iou_np)

        pred_boxes, pred_scores, gt_boxes = example
        gt_ignore = np.array([1, 0, 0])
        iou_thresholds = [0.1, 0.5]

        res = matching._matching_single_image_single_class(
            iou_thresholds=iou_thresholds,
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            gt_boxes=gt_boxes,
            gt_ignore=gt_ignore,
            case_id="example",
        )

        assert np.isclose(res["dtMatches"], [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]]).all()
        assert np.isclose(res["gtMatches"], [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]).all()
        assert np.isclose(res["dtScores"], [0.9, 0.5, 0.3]).all()
        assert np.isclose(res["gtIgnore"], [0.0, 0.0, 1.0]).all()
        assert np.isclose(res["dtIgnore"], [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).all()

    def test_box_matching_single_image_single_class_mul_matches(self, example):
        matching = EvalMatchingPerElementGreedyScoreNP(iou_fn=ops_np.box_iou_np)

        pred_boxes, pred_scores, gt_boxes = example
        pred_boxes[0, :] = pred_boxes[2, :]  # simulate duplicate prediction
        gt_ignore = np.array([0, 0, 0])
        iou_thresholds = [0.2, 0.5]

        res = matching._matching_single_image_single_class(
            iou_thresholds=iou_thresholds,
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            gt_boxes=gt_boxes,
            gt_ignore=gt_ignore,
            case_id="example",
        )

        assert np.isclose(res["dtMatches"], [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]).all()
        assert np.isclose(res["gtMatches"], [[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]).all()
        assert np.isclose(res["dtScores"], [0.9, 0.5, 0.3]).all()
        assert np.isclose(res["gtIgnore"], [0.0, 0.0, 0.0]).all()
        assert np.isclose(res["dtIgnore"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).all()

    def test_matching_no_gt(self, example):
        matching = EvalMatchingPerElementGreedyScoreNP(iou_fn=ops_np.box_iou_np)

        _, pred_scores, _ = example
        iou_thresholds = [0.1, 0.5]

        res = matching._matching_no_gt(
            iou_thresholds=iou_thresholds,
            pred_scores=pred_scores,
            case_id="example",
        )

        assert np.isclose(res["dtMatches"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).all()
        assert res["gtMatches"].size == 0
        assert res["gtMatches"].shape == (2, 0)
        assert np.isclose(res["dtScores"], [0.9, 0.5, 0.3]).all()
        assert res["gtIgnore"].size == 0
        assert res["gtIgnore"].shape == (0,)
        assert np.isclose(res["dtIgnore"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).all()

    def test_matching_no_pred(self):
        matching = EvalMatchingPerElementGreedyScoreNP(iou_fn=ops_np.box_iou_np)

        iou_thresholds = [0.1, 0.5]
        gt_ignore = np.array([0.0, 0.0, 1.0])

        res = matching._matching_no_pred(
            iou_thresholds=iou_thresholds,
            gt_ignore=gt_ignore,
            case_id="example",
        )

        assert res["dtMatches"].size == 0
        assert res["dtMatches"].shape == (2, 0)
        assert np.isclose(res["gtMatches"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).all()
        assert res["dtScores"].size == 0
        assert res["dtScores"].shape == (0,)
        assert np.isclose(res["gtIgnore"], [0.0, 0.0, 1.0]).all()
        assert res["dtIgnore"].size == 0
        assert res["dtIgnore"].shape == (2, 0)

    def test_smoke_box_matching_batch(self):
        matching = EvalMatchingPerElementGreedyScoreNP(iou_fn=ops_np.box_iou_np)

        _pd_boxes = [
            np.array(
                [
                    [0.0, 0.0, 10.0, 10.0],
                    [2.0, 2.0, 10.0, 10.0],
                    [20.0, 20.0, 30.0, 30.0],
                ]
            )
        ]
        _pd_classes = [np.array([0, 1, 1])]
        _pd_scores = [np.array([0.9, 0.5, 0.6])]
        _gt_boxes = [
            np.array(
                [
                    [0.0, 0.0, 10.0, 10.0],
                    [2.0, 2.0, 10.0, 10.0],
                    [20.0, 20.0, 30.0, 30.0],
                    [30.0, 30.0, 40.0, 40.0],
                ]
            )
        ]
        _gt_classes = [np.array([1, 1, 1, 0])]
        _gt_ignore = [np.array([0, 0, 0, 0])]
        matching.match(
            [0.1, 0.5, 0.75],
            _pd_boxes,
            _pd_classes,
            _pd_scores,
            _gt_boxes,
            _gt_classes,
            _gt_ignore,
        )
