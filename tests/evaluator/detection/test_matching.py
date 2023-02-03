import numpy as np
import pytest
from pytest_mock import MockerFixture

import nndet.core.ops_np as ops_np
from nndet.evaluator.detection.matching import (
    _matching_no_gt,
    _matching_no_pred,
    _matching_single_image_single_class,
    matching_batch,
)


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
        box_match_single_mock = mocker.patch(
            "nndet.evaluator.detection.matching._matching_single_image_single_class",
            return_value=0,
        )
        iou_fn = mocker.Mock(return_value=None)
        iou_thresholds = [0.1, 0.5]

        pred_boxes = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
        pred_classes = np.array([1, 0])
        pred_scores = np.array([1.0, 1.0])

        gt_boxes = np.array([[4, 5, 6, 7], [4, 5, 6, 7]])
        gt_classes = np.array([0, 1])
        gt_ignore = np.array([0, 0])

        res = matching_batch(
            iou_fn,
            iou_thresholds,
            [pred_boxes],
            [pred_classes],
            [pred_scores],
            [gt_boxes],
            [gt_classes],
            [gt_ignore],
        )

        assert len(res) == 1
        assert {0: 0, 1: 0} == res[0]
        box_match_single_mock.assert_called()

    def test_box_matching_batch_no_gt(self, mocker: MockerFixture):
        match_fn_mocker = mocker.patch(
            "nndet.evaluator.detection.matching._matching_no_gt",
            return_value=0,
        )
        iou_fn = mocker.Mock(return_value=None)
        iou_thresholds = [0.1, 0.5]

        pred_boxes = np.array([[0.0, 1.0, 2.0, 3.0]])
        pred_classes = np.array([0])
        pred_scores = np.array([1.0])

        gt_boxes = np.array([[]])
        gt_classes = np.array([])
        gt_ignore = np.array([])

        res = matching_batch(
            iou_fn,
            iou_thresholds,
            [pred_boxes],
            [pred_classes],
            [pred_scores],
            [gt_boxes],
            [gt_classes],
            [gt_ignore],
        )

        assert len(res) == 1
        assert res == [{0: 0}]
        match_fn_mocker.assert_called()

    def test_box_matching_batch_no_pred(self, mocker: MockerFixture):
        match_fn_mocker = mocker.patch(
            "nndet.evaluator.detection.matching._matching_no_pred",
            return_value=0,
        )
        iou_fn = mocker.Mock(return_value=None)
        iou_thresholds = [0.1, 0.5]

        pred_boxes = np.array([[]])
        pred_classes = np.array([])
        pred_scores = np.array([])

        gt_boxes = np.array([[0.0, 1.0, 2.0, 3.0]])
        gt_classes = np.array([0])
        gt_ignore = np.array([1.0])

        res = matching_batch(
            iou_fn,
            iou_thresholds,
            [pred_boxes],
            [pred_classes],
            [pred_scores],
            [gt_boxes],
            [gt_classes],
            [gt_ignore],
        )

        assert len(res) == 1
        assert res == [{0: 0}]
        match_fn_mocker.assert_called()

    def test_box_matching_single_image_single_class_no_ignore(self, example):
        pred_boxes, pred_scores, gt_boxes = example
        gt_ignore = np.array([0, 0, 0])
        iou_thresholds = [0.1, 0.5]

        res = _matching_single_image_single_class(
            iou_fn=ops_np.box_iou_np,
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            gt_boxes=gt_boxes,
            gt_ignore=gt_ignore,
            max_detections=100,
            iou_thresholds=iou_thresholds,
            case_id="example",
        )

        assert np.isclose(res["dtMatches"], [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]]).all()
        assert np.isclose(res["gtMatches"], [[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]).all()
        assert np.isclose(res["dtScores"], [0.9, 0.5, 0.3]).all()
        assert np.isclose(res["gtIgnore"], [0.0, 0.0, 0.0]).all()
        assert np.isclose(res["dtIgnore"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).all()
        assert res["case_id"] == "example"

    def test_box_matching_single_image_single_class_ignore(self, example):
        pred_boxes, pred_scores, gt_boxes = example
        gt_ignore = np.array([1, 0, 0])
        iou_thresholds = [0.1, 0.5]

        res = _matching_single_image_single_class(
            iou_fn=ops_np.box_iou_np,
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            gt_boxes=gt_boxes,
            gt_ignore=gt_ignore,
            max_detections=100,
            iou_thresholds=iou_thresholds,
            case_id="example",
        )

        assert np.isclose(res["dtMatches"], [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]]).all()
        assert np.isclose(res["gtMatches"], [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]).all()
        assert np.isclose(res["dtScores"], [0.9, 0.5, 0.3]).all()
        assert np.isclose(res["gtIgnore"], [0.0, 0.0, 1.0]).all()
        assert np.isclose(res["dtIgnore"], [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).all()
        assert res["case_id"] == "example"

    def test_box_matching_single_image_single_class_mul_matches(self, example):
        pred_boxes, pred_scores, gt_boxes = example
        pred_boxes[0, :] = pred_boxes[2, :]  # simulate duplicate prediction
        gt_ignore = np.array([0, 0, 0])
        iou_thresholds = [0.2, 0.5]

        res = _matching_single_image_single_class(
            iou_fn=ops_np.box_iou_np,
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            gt_boxes=gt_boxes,
            gt_ignore=gt_ignore,
            max_detections=100,
            iou_thresholds=iou_thresholds,
            case_id="example",
        )

        assert np.isclose(res["dtMatches"], [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]).all()
        assert np.isclose(res["gtMatches"], [[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]).all()
        assert np.isclose(res["dtScores"], [0.9, 0.5, 0.3]).all()
        assert np.isclose(res["gtIgnore"], [0.0, 0.0, 0.0]).all()
        assert np.isclose(res["dtIgnore"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).all()
        assert res["case_id"] == "example"

    def test_matching_no_gt(self, example):
        _, pred_scores, _ = example
        iou_thresholds = [0.1, 0.5]

        res = _matching_no_gt(
            iou_thresholds=iou_thresholds,
            pred_scores=pred_scores,
            max_detections=100,
            case_id="example",
        )

        assert np.isclose(res["dtMatches"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).all()
        assert res["gtMatches"].size == 0
        assert res["gtMatches"].shape == (2, 0)
        assert np.isclose(res["dtScores"], [0.9, 0.5, 0.3]).all()
        assert res["gtIgnore"].size == 0
        assert res["gtIgnore"].shape == (0,)
        assert np.isclose(res["dtIgnore"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).all()
        assert res["case_id"] == "example"

    def test_matching_no_pred(self, example):
        iou_thresholds = [0.1, 0.5]
        gt_ignore = np.array([0.0, 0.0, 1.0])

        res = _matching_no_pred(
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
        assert res["case_id"] == "example"

    def test_smoke_box_matching_batch(self):
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
        matching_batch(
            ops_np.box_iou_np,
            [0.1, 0.5, 0.75],
            _pd_boxes,
            _pd_classes,
            _pd_scores,
            _gt_boxes,
            _gt_classes,
            _gt_ignore,
        )
