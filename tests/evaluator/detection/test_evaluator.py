import json

import numpy as np
import pytest
from pytest_mock import MockerFixture

import nndet.core.ops_np as ops_np
from nndet.evaluator.det import BoxEvaluator, DetectionEvaluator
from nndet.evaluator.detection.coco import COCOMetric


class DummyMetric:
    def __init__(self, ious=(0.1, 0.2)):
        self.ious = ious

    def get_iou_thresholds(self):
        return self.ious


@pytest.fixture
def evaluator():
    return DetectionEvaluator([DummyMetric()], iou_fn=ops_np.box_iou_np)


class TestDetectionEvaluator:
    def test_init(self):
        self.evaluator = DetectionEvaluator(
            [
                DummyMetric((0.1, 0.2)),
                DummyMetric((0.3, 0.4)),
            ],
            iou_fn=ops_np.box_iou_np,
        )
        assert all([a == b for a, b in zip(self.evaluator.iou_thresholds, [0.1, 0.2, 0.3, 0.4])])
        assert all([a == b for a, b in zip(self.evaluator.iou_mapping, [[0, 1], [2, 3]])])
        assert "" in self.evaluator.criterion_ranges.keys()
        assert self.evaluator.criterion_ranges[""][0] == np.NINF
        assert self.evaluator.criterion_ranges[""][1] == np.inf

    def test_run_online_evaluation(self, mocker: MockerFixture, evaluator):
        _pred_boxes = np.array([[0]])[None]
        _pred_classes = np.array([[1]])[None]
        _pred_scores = np.array([[2]])[None]
        _gt_boxes = np.array([[3]])[None]
        _gt_classes = np.array([[4]])[None]
        # Use pred and gt class here
        mock_matches = {
            1: {"dtMatches": np.array([[1, 1]]), "dtIgnore": np.array([[0, 0]])},
            4: {"dtMatches": np.array([[1, 1]]), "dtIgnore": np.array([[0, 0]])},
        }
        evaluator.match_fn = mocker.MagicMock(return_value=[mock_matches])
        evaluator.box_criterion = mocker.MagicMock(return_value=[0])
        res = evaluator.run_online_evaluation(
            _pred_boxes,
            _pred_classes,
            _pred_scores,
            _gt_classes,
            _gt_classes,
        )

        assert not res
        assert all(a == b for a, b in zip([""], evaluator.results_dict.keys()))
        assert len(evaluator.results_dict[""]) == 1
        assert all(mock_matches[key] == value for key, value in evaluator.results_dict[""][0].items())

    def test_find_dt_ignores(self, mocker: MockerFixture, evaluator):
        _pred_boxes = np.array([[0]])[None]
        _pred_classes = np.array([[1]])[None]
        _pred_scores = np.array([[2]])[None]
        _gt_boxes = np.array([[3]])[None]
        _gt_classes = np.array([[1]])[None]
        _gt_ignore = np.array([[0]])[None]
        # Use pred and gt class here (1), has to be unmatched so dtMatch 0
        mock_matches = {1: {"dtMatches": np.array([[0]]), "dtIgnore": np.array([[0]])}}
        # Match should be ignored as criterion returns 2 but bounds are (0, 1)
        evaluator.box_criterion = mocker.MagicMock(return_value=[2])
        evaluator.criterion_ranges[""] = (0, 1)
        # List[Dict[class, Dict]]
        matches_with_ignores = evaluator.find_dt_ignores(
            results_key="",
            matches=[mock_matches],
            iou_thresholds=[1],
            pred_boxes=_pred_boxes,
            pred_classes=_pred_classes,
            pred_scores=_pred_scores,
            gt_boxes=_gt_boxes,
            gt_classes=_gt_classes,
            gt_ignore=_gt_ignore,
        )

        assert len(matches_with_ignores) == 1
        res = matches_with_ignores[0]
        assert all(res[c]["dtIgnore"][i] == 1 for c in res.keys() for i in range(len(res[c]["dtIgnore"])))

    def test_finish_online_evaluation(self, mocker: MockerFixture, evaluator):
        evaluator.iou_filter = mocker.Mock(return_value=0)
        metric0 = mocker.Mock(return_value=({"score0": 0}, {"curve0": 1}))
        metric1 = mocker.Mock(return_value=({"score1": 2}, {"curve1": 3}))

        evaluator.metrics = [metric0, metric1]
        evaluator.results_dict = {"": [None, None]}
        evaluator.iou_mapping = [[0], [1]]
        metric_scores, metric_curves = evaluator.finish_online_evaluation()

        assert metric_curves == {"curve0": 1, "curve1": 3, "criterion": (np.NINF, np.inf)}
        assert metric_scores == {"score0": 0, "score1": 2}
        metric0.assert_called_with([0, 0], tag="")
        metric1.assert_called_with([0, 0], tag="")

    def test_iou_filter(self, evaluator):
        image_dict = {
            0: {"dtMatches": np.array([0, 1, 2, 3])},
            1: {"dtMatches": np.array([2, 3, 0, 1])},
        }
        res = evaluator.iou_filter(image_dict, iou_idx=[0, 1], filter_keys=["dtMatches"])
        assert np.isclose(res[0]["dtMatches"], [0, 1]).all()
        assert np.isclose(res[1]["dtMatches"], [2, 3]).all()
