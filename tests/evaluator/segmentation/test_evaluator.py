import pytest
import numpy as np

from nndet.evaluator.seg import SegmentationEvaluator


@pytest.fixture
def evaluator():
    return SegmentationEvaluator()


@pytest.fixture
def target():
    target = np.ones((1, 30, 10))
    target[:, 10:] = 2.0
    return target


@pytest.fixture
def pred():
    pred = np.zeros((1, 3, 30, 10))
    pred[:, 1] += 0.5
    pred[:, 2, 0:20] = 0.6
    return pred


class TestSegmentationEvaluator:
    def test_run_online_evaluation_tp(self, evaluator, pred, target):
        evaluator.run_online_evaluation(pred, target)
        assert len(evaluator.results_list["tp"]) == 1
        assert np.allclose(evaluator.results_list["tp"][0], np.array([0.0, 100.0]))
        assert len(evaluator.results_list["fp"]) == 1
        assert np.allclose(evaluator.results_list["fp"][0], np.array([100.0, 100.0]))
        assert len(evaluator.results_list["fn"]) == 1
        assert np.allclose(evaluator.results_list["fn"][0], np.array([100.0, 100.0]))

        evaluator.reset()
        assert not evaluator.results_list

    def test_finish_online_evaluation(self, evaluator, pred, target):
        evaluator.run_online_evaluation(pred, target)
        evaluator.run_online_evaluation(pred, target)
        seg_scores, _ = evaluator.finish_online_evaluation()
        assert seg_scores["0_seg_dice"] == 0.0
        assert seg_scores["1_seg_dice"] == 0.5
        assert seg_scores["seg_dice"] == 0.25

    def test_finish_online_evaluation_empty(self, evaluator):
        seg_scores, _ = evaluator.finish_online_evaluation()
        assert not seg_scores
