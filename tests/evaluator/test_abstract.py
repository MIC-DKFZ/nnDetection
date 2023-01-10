import numpy as np
import pytest

from nndet.evaluator import AbstractEvaluator, AbstractMetric, DetectionMetric


class DummyEvaluator(AbstractEvaluator):
    def run_online_evaluation(self):
        super().run_online_evaluation()

    def finish_online_evaluation(self):
        super().finish_online_evaluation()

    def reset(self):
        pass


class DummyDetectionMetric(DetectionMetric):
    def compute(self, tmp):
        super().compute(None)

    def get_iou_thresholds(self):
        super().get_iou_thresholds()

    def get_save_name(self):
        super().get_save_name()


class TestAbstractEvaluator:
    def test_abstract_evaluator_instance(self):
        with pytest.raises(TypeError):
            evaluator = AbstractEvaluator()

    def test_dummy_evaluator_run_online_evaluation(self):
        evaluator = DummyEvaluator()
        with pytest.raises(NotImplementedError):
            evaluator.run_online_evaluation()

    def test_dummy_evaluator_finish_online_evaluation(self):
        evaluator = DummyEvaluator()
        with pytest.raises(NotImplementedError):
            evaluator.finish_online_evaluation()


class TestDetectionMetric:
    def test_abstract_metric_compute(self):
        metric = DummyDetectionMetric()
        with pytest.raises(NotImplementedError):
            metric.compute(None)

    def test_detection_metric_call(self):
        metric = DummyDetectionMetric()
        with pytest.raises(NotImplementedError):
            metric(None)

    def test_detection_metric_get_iou_thresholds(self):
        metric = DummyDetectionMetric()
        with pytest.raises(NotImplementedError):
            metric.get_iou_thresholds()

    def test_detection_metric_check_number_of_iou(self):
        metric = DummyDetectionMetric()
        metric.get_iou_thresholds = lambda: [0.1, 0.5]

        metric.check_number_of_iou(np.zeros((2, 100)))

        with pytest.raises(AssertionError):
            metric.check_number_of_iou(np.zeros((2, 100)), np.zeros((1, 100)))
