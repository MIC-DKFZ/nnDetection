import numpy as np
import pytest
from pytest_mock import MockerFixture

import nndet.core.ops_np as ops_np
from nndet.eval.det import BoxEvaluator
from nndet.eval.matching import EvalMatchingPerElementGreedyScoreNP


class DummyMetric:
    def __init__(self, ious=(0.1, 0.2)):
        self.ious = ious

    def get_iou_thresholds(self):
        return self.ious


@pytest.fixture
def evaluator():
    matching = EvalMatchingPerElementGreedyScoreNP(
        iou_fn=ops_np.box_iou_np,
        max_detections=100,
    )
    return BoxEvaluator([DummyMetric()], matching=matching)


class TestBoxEvaluator:
    def test_init(self):
        matching = EvalMatchingPerElementGreedyScoreNP(
            iou_fn=ops_np.box_iou_np,
            max_detections=100,
        )
        evaluator = BoxEvaluator(
            [
                DummyMetric((0.1, 0.2)),
                DummyMetric((0.3, 0.4)),
            ],
            matching=matching,
        )
        assert all([a == b for a, b in zip(evaluator.iou_thresholds, [0.1, 0.2, 0.3, 0.4])])
        assert all([a == b for a, b in zip(evaluator.iou_mapping, [[0, 1], [2, 3]])])
        assert "" in evaluator.criterion_ranges.keys()
        assert evaluator.criterion_ranges[""][0] == -np.inf
        assert evaluator.criterion_ranges[""][1] == np.inf

    def test_run_online_evaluation_smoke(self, evaluator):
        _pred_boxes = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])[None]
        _pred_classes = np.array([1])
        _pred_scores = np.array([0.1])
        _gt_boxes = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])[None]
        _gt_classes = np.array([2])
        res = evaluator.run_online_evaluation(
            [_pred_boxes],
            [_pred_classes],
            [_pred_scores],
            [_gt_boxes],
            [_gt_classes],
        )
        assert not res

    def test_run_online_evaluation(self, mocker: MockerFixture, evaluator):
        mock_matches = {
            1: {"dtMatches": np.array([[1, 1]]), "dtIgnore": np.array([[0, 0]])},
            4: {"dtMatches": np.array([[1, 1]]), "dtIgnore": np.array([[0, 0]])},
        }
        evaluator.matching.match = mocker.MagicMock(return_value=mock_matches)
        evaluator.criterion = mocker.MagicMock(return_value=np.array([0]))

        _pred_boxes = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])[None]
        _pred_classes = np.array([1])
        _pred_scores = np.array([0.1])
        _gt_boxes = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])[None]
        _gt_classes = np.array([2])
        res = evaluator.run_online_evaluation(
            [_pred_boxes],
            [_pred_classes],
            [_pred_scores],
            [_gt_boxes],
            [_gt_classes],
        )

        assert not res
        assert all(a == b for a, b in zip([""], evaluator.results_dict.keys()))
        assert len(evaluator.results_dict[""]) == 1
        assert all(mock_matches[key] == value for key, value in evaluator.results_dict[""][0].items())

    def test_finish_online_evaluation(self, mocker: MockerFixture, evaluator):
        evaluator.iou_filter = mocker.Mock(return_value=0)
        metric0 = mocker.Mock(return_value=({"score0": 0}, {"curve0": 1}))
        metric1 = mocker.Mock(return_value=({"score1": 2}, {"curve1": 3}))

        evaluator.metrics = [metric0, metric1]
        evaluator.results_dict = {"": {0: None, 1: None}}
        evaluator.iou_mapping = [[0], [1]]
        metric_scores, metric_curves = evaluator.finish_online_evaluation()

        metric_curves.pop("__eval")
        assert metric_curves == {"curve0": 1, "curve1": 3}
        assert metric_scores == {"score0": 0, "score1": 2}
        metric0.assert_called_with([0, 0], tag=None)
        metric1.assert_called_with([0, 0], tag=None)

    def test_iou_filter(self, evaluator):
        image_dict = {
            0: {"dtMatches": np.array([0, 1, 2, 3])},
            1: {"dtMatches": np.array([2, 3, 0, 1])},
        }
        res = evaluator.iou_filter(image_dict, iou_idx=[0, 1], filter_keys=["dtMatches"])
        assert np.isclose(res[0]["dtMatches"], [0, 1]).all()
        assert np.isclose(res[1]["dtMatches"], [2, 3]).all()
