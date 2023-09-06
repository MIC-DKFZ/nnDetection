import math

import numpy as np
import pytest
from pytest_mock import MockerFixture

from nndet.eval.det.coco import COCOMetric, compute_stats_single_threshold


@pytest.fixture
def metric():
    return COCOMetric(
        classes=["benign", "malignant"],
        iou_list=(0.1, 0.3),
        iou_range=(0.1, 0.2, 0.1),
        max_detection=(1, 10),
    )


class TestCOCOMetric:
    def test_get_iou_thresholds(self):
        metric = COCOMetric(
            classes=["benign", "malignant"],
            iou_list=(0.1, 0.2, 0.3),
            iou_range=(0.1, 0.2, 0.05),
            max_detection=(1, 5, 100),
        )
        assert np.isclose(metric.get_iou_thresholds(), [0.1, 0.15, 0.2, 0.3]).all()

    def test_compute(self, mocker: MockerFixture, metric):
        mocker.patch("nndet.evaluator.detection.coco.COCOMetric.select_ap", return_value=1)
        mocker.patch(
            "nndet.evaluator.detection.coco.COCOMetric.compute_statistics",
            return_value={"stats": 0},
        )

        score, curve = metric([0, 1, 2])
        assert curve is None
        assert score["mAP_IoU_0.10_0.20_0.10_MaxDet_10"] == 1
        assert score["AP_IoU_0.10_MaxDet_10"] == 1
        assert score["AP_IoU_0.30_MaxDet_10"] == 1

    def test_select_ap(self, metric):
        stats = {"precision": np.array([[0.0, 0.5, 1.0], [1.0, 1.0, 1.0], [1.0, 0.5, 3.0]])[:, :, None, None]}
        ap0 = metric.select_ap(stats, [0])
        ap = metric.select_ap(stats)
        assert math.isclose(ap0, 0.5)
        assert math.isclose(ap, 1.0)

    def test_compute_statistics(self, mocker: MockerFixture, metric):
        metric.iou_thresholds = np.array([[0.1]])
        metric.recall_thresholds = np.array([0.1, 0.2])
        results_list = []
        results_list += [
            {
                0: {
                    "dtMatches": np.array([[0, 0]]),
                    "dtIgnore": np.array([[0, 0]]),
                    "dtScores": np.array([0, 0]),
                    "gtIgnore": np.array([0]),
                },
                1: {
                    "dtMatches": np.array([[0, 0]]),
                    "dtIgnore": np.array([[0, 0]]),
                    "dtScores": np.array([0, 0]),
                    "gtIgnore": np.array([0]),
                },
            }
        ] * 3
        mocker.patch(
            "nndet.evaluator.detection.coco.compute_stats_single_threshold",
            return_value=(1, [2, 3], [4, 5]),
        )

        stats = metric.compute_statistics(results_list)
        assert np.isclose(stats["counts"], [1, 2, 2, 2]).all()
        assert np.isclose(stats["recall"], [[[1.0, 1.0], [1.0, 1.0]]]).all()
        assert np.isclose(stats["precision"], [[[[2.0, 2.0], [2.0, 2.0]], [[3.0, 3.0], [3.0, 3.0]]]]).all()
        assert np.isclose(stats["scores"], [[[[4.0, 4.0], [4.0, 4.0]], [[5.0, 5.0], [5.0, 5.0]]]]).all()

    def test_compute_stats_single_threshold(self):
        tp = np.array([1, 2, 3, 4, 5, 6])
        fp = np.array([0, 1, 1, 2, 3, 3])
        dt_scores_sorted = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        recall_thresholds = np.array([0.5, 0.7, 0.9])
        num_gt = 2
        rc, prec, ths = compute_stats_single_threshold(tp, fp, dt_scores_sorted, recall_thresholds, num_gt)

        assert math.isclose(rc, 3.0)  # used different num_gt so number are nice
        assert np.isclose(prec, [1.0, 0.75, 0.75]).all()
        assert np.isclose(ths, [0.9, 0.8, 0.8]).all()
