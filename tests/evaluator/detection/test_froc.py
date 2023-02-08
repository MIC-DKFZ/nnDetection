import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import call

import numpy as np
import pytest
from pytest_mock import MockerFixture

from nndet.evaluator.detection import FROCMetric


@pytest.fixture
def metric():
    return FROCMetric(
        ["benign", "malignant"],
        iou_thresholds=[0.1],
        # due to wrong defaults in an earilier version, the tests use these values
        fpi_thresholds=(1 / 8, 1 / 4, 1 / 2, 2, 4, 8),
    )


@pytest.fixture
def results_list():
    results_list = []
    results_list += [{0: {"dtMatches": 0}}] * 3
    results_list += [{1: {"dtMatches": 1}}] * 3
    return results_list


class TestFROC:
    def test_get_iou_thresholds(self, metric):
        assert metric.get_iou_thresholds() == [0.1]

    def test_compute(self, mocker: MockerFixture, metric, results_list):
        froc_mul_class_mock = mocker.Mock(return_value=({"froc_score_cls": 0}, {"froc_curve_cls": 1}))
        metric.compute_froc_mul_iou_per_class = froc_mul_class_mock

        froc_mul_iou_mock = mocker.Mock(return_value=({"froc_score": 1}, {"froc_curve": 2}))
        metric.compute_froc_mul_iou = froc_mul_iou_mock

        froc_score, froc_curve = metric(results_list)
        for key in ["iou_thresholds", "fpi_thresholds", "classes"]:
            assert key in froc_curve
            froc_curve.pop(key)

        assert {"froc_score": 1, "froc_score_cls": 0} == froc_score
        assert {"froc_curve": 2, "froc_curve_cls": 1} == froc_curve

        froc_mul_class_mock.assert_called_once()
        froc_mul_iou_mock.assert_called_once()

    def test_compute_froc_mul_iou_no_gt(self, mocker: MockerFixture, metric):
        mock = mocker.Mock()
        metric.compute_froc_curve_one_iou = mock

        results_list = []
        results_list += [
            {
                0: {
                    "dtMatches": np.array([[0]]),
                    "dtIgnore": np.array([[0]]),
                    "dtScores": np.array([0]),
                    "gtIgnore": np.array([1]),
                }
            }
        ] * 3
        froc_score, froc_curve = metric(results_list)

        for key, item in froc_score.items():
            assert np.isnan(item).all()

        for key in ["iou_thresholds", "fpi_thresholds", "classes"]:
            assert key in froc_curve
            froc_curve.pop(key)

        # no class
        assert froc_curve.pop("num_images") == 3
        assert froc_curve.pop("num_gt") == 0

        # benign
        assert froc_curve.pop("benign_num_images") == 3
        assert froc_curve.pop("benign_num_gt") == 0

        # malignant
        assert froc_curve.pop("malignant_num_images") == 3
        assert froc_curve.pop("malignant_num_gt") == 0

        for key, item in froc_curve.items():
            assert np.isclose(item, 0).all()

    def test_compute_froc_mul_iou(self, mocker: MockerFixture, metric):
        mock = mocker.Mock(return_value=([0, 1], [0, 1], 0))
        metric.compute_froc_curve_one_iou = mock

        results_list = []
        results_list += [
            {
                0: {
                    "dtMatches": np.array([[0]]),
                    "dtIgnore": np.array([[0]]),
                    "dtScores": np.array([0]),
                    "gtIgnore": np.array([0]),
                }
            }
        ] * 3

        froc_score, froc_curve = metric(results_list)
        assert math.isclose(3.875 / 6, froc_score["FROC_IoU_0.10"])
        assert math.isclose(3.875 / 6, froc_score["benign_FROC_IoU_0.10"])
        assert np.isnan(froc_score["mc_FROC_IoU_0.10"]).all()
        assert np.isclose(
            froc_curve["FROC_IoU_0.10"],
            np.array([0.125, 0.25, 0.5, 1.0, 1.0, 1.0]),
        ).all()

    def test_compute_froc_mul_iou_per_class(self, mocker: MockerFixture, metric, results_list):
        froc_mul_iou_mock = mocker.Mock(return_value=({"froc_score": 1}, {"froc_curve": 2}))
        metric.compute_froc_mul_iou = froc_mul_iou_mock

        froc_score, froc_curve = metric.compute_froc_mul_iou_per_class(results_list, tag=None)

        assert {
            "benign_froc_score": 1,
            "malignant_froc_score": 1,
            "mc_froc_score": 1,
        } == froc_score
        assert {"benign_froc_curve": 2, "malignant_froc_curve": 2} == froc_curve

        froc_mul_iou_mock.assert_has_calls(
            [
                call([{0: {"dtMatches": 0}}] * 3 + [{}] * 3, tag=None),
                call([{}] * 3 + [{0: {"dtMatches": 1}}] * 3, tag=None),
            ]
        )

    def test_compute_froc_curve_one_iou(self, metric):
        _dt_matches = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        _dt_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.85, 0.75, 0.65, 0.55])
        num_images = 4
        num_gt = 4
        fps, sens, th = metric.compute_froc_curve_one_iou(_dt_matches, _dt_scores, num_images, num_gt)
        assert np.isclose(fps, [0.0, 0.0, 1.0 / 4, 1.0 / 4, 1.0 / 2, 1.0 / 2, 3.0 / 4, 3.0 / 4, 1.0]).all()
        assert np.isclose(sens, [0.0, 1.0 / 4, 1.0 / 4, 1.0 / 2, 1.0 / 2, 3.0 / 4, 3.0 / 4, 1.0, 1.0]).all()
        assert np.isclose(th[1:], [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55]).all()
