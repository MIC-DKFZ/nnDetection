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
        froc_mul_class_mock = mocker.Mock(
            return_value=({"froc_score_cls": 0}, {"froc_curve_cls": 1})
        )
        metric.compute_froc_mul_iou_per_class = froc_mul_class_mock

        froc_mul_iou_mock = mocker.Mock(
            return_value=({"froc_score": 1}, {"froc_curve": 2})
        )
        metric.compute_froc_mul_iou = froc_mul_iou_mock

        froc_score, froc_curve = metric(results_list)
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
            assert math.isclose(item, 0)

        # no class
        froc_curve.pop("FROC_fpi_thresholds")
        assert froc_curve.pop("FROC_num_images") == 3
        assert froc_curve.pop("FROC_num_gt") == 0

        # benign
        froc_curve.pop("benign_FROC_fpi_thresholds")
        assert froc_curve.pop("benign_FROC_num_images") == 3
        assert froc_curve.pop("benign_FROC_num_gt") == 0

        # malignant
        froc_curve.pop("malignant_FROC_fpi_thresholds")
        assert froc_curve.pop("malignant_FROC_num_images") == 3
        assert froc_curve.pop("malignant_FROC_num_gt") == 0

        for key, item in froc_curve.items():
            assert np.isclose(item, 0).all()

    def test_compute_froc_mul_iou(self, mocker: MockerFixture, metric):
        mock = mocker.Mock(return_value=([0, 1], [0, 1], 0))
        metric.compute_froc_curve_one_iou = mock
        metric.per_class = False

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
        assert {"FROC_score_IoU_0.10": 3.875 / 6} == froc_score
        assert np.isclose(
            froc_curve["FROC_curve_IoU_0.10"],
            np.array([0.125, 0.25, 0.5, 1.0, 1.0, 1.0]),
        ).all()

    def test_compute_froc_mul_iou_per_class(
        self, mocker: MockerFixture, metric, results_list
    ):
        froc_mul_iou_mock = mocker.Mock(
            return_value=({"froc_score": 1}, {"froc_curve": 2})
        )
        metric.compute_froc_mul_iou = froc_mul_iou_mock

        froc_score, froc_curve = metric.compute_froc_mul_iou_per_class(results_list)

        assert {
            "benign_froc_score": 1,
            "malignant_froc_score": 1,
            "mc_froc_score": 1,
        } == froc_score
        assert {"benign_froc_curve": 2, "malignant_froc_curve": 2} == froc_curve

        froc_mul_iou_mock.assert_has_calls(
            [
                call([{0: {"dtMatches": 0}}] * 3 + [{}] * 3),
                call([{}] * 3 + [{0: {"dtMatches": 1}}] * 3),
            ]
        )

    def test_compute_froc_curve_one_iou(self, metric):
        _dt_matches = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        _dt_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.85, 0.75, 0.65, 0.55])
        num_images = 4
        num_gt = 4
        fps, sens, th = metric.compute_froc_curve_one_iou(
            _dt_matches, _dt_scores, num_images, num_gt
        )
        assert np.isclose(
            fps, [0.0, 0.0, 1.0 / 4, 1.0 / 4, 1.0 / 2, 1.0 / 2, 3.0 / 4, 3.0 / 4, 1.0]
        ).all()
        assert np.isclose(
            sens, [0.0, 1.0 / 4, 1.0 / 4, 1.0 / 2, 1.0 / 2, 3.0 / 4, 3.0 / 4, 1.0, 1.0]
        ).all()
        assert np.isclose(th[1:], [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55]).all()

    def test_froc_plotting(self, metric):
        with TemporaryDirectory(dir=os.getcwd()) as _dir:
            vals = np.array([0.0, 1.0 / 4, 1.0 / 4, 1.0 / 2, 3.0 / 4, 1.0])

            frocs = {
                f"FROC_curve_IoU_{iou:.2f}": vals + iou / 10 for iou in range(0, 10)
            }
            frocs[f"mal_FROC_curve_IoU_{0.1:.2f}"] = [
                0.0,
                1.0 / 4,
                1.0 / 4,
                1.0 / 2,
                3.0 / 4,
                1.0,
            ]
            frocs[f"ben_FROC_curve_IoU_{0.1:.2f}"] = [
                0.1,
                1.0 / 8,
                1.0 / 2,
                1.0 / 2,
                3.0 / 4,
                1.0,
            ]
            frocs[f"mal_FROC_curve_IoU_{0.2:.2f}"] = [
                0.0,
                1.0 / 4,
                1.0 / 4,
                1.0 / 2,
                3.0 / 4,
                1.0,
            ]
            frocs[f"ben_FROC_curve_IoU_{0.2:.2f}"] = [
                0.1,
                1.0 / 8,
                1.0 / 2,
                1.0 / 2,
                3.0 / 4,
                1.0,
            ]

            frocs["FROC_num_images"] = 10
            frocs["mal_FROC_num_images"] = 10
            frocs["ben_FROC_num_images"] = 10

            frocs["FROC_num_gt"] = 10
            frocs["mal_FROC_num_gt"] = 10
            frocs["ben_FROC_num_gt"] = 10
            metric.save_dir = Path(_dir)
            metric.plot_froc_curves(frocs)
