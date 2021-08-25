import pytest
from pytest_mock import MockerFixture

import numpy as np
from unittest.mock import call

from nndet.evaluator.detection.matching import matching_batch
from nndet.core.boxes import box_iou_np


class TestMatching:
    def test_box_matching_batch(self, mocker: MockerFixture):
        box_match_single_mock = mocker.patch(
            "nndet.evaluator.detection.matching._matching_single_image_single_class",
            return_value=0,
        )

        iou_fn = mocker.Mock(return_value=None)
        iou_tresholds = [0.1, 0.5]

        pred_boxes = np.array([[0, 1, 2, 3]])
        pred_classes = np.array([[1, 0, 1, 2]])
        pred_scores = np.array([[1.0, 0.9, 0.8, 0.7]])

        gt_boxes = np.array([[4, 5, 6, 7]])
        gt_classes = np.array([[1, 2, 0, 0]])
        gt_ignore = np.array([[0, 1, 0, 0]])

        res = matching_batch(
            iou_fn,
            iou_tresholds,
            pred_boxes,
            pred_classes,
            pred_scores,
            gt_boxes,
            gt_classes,
            gt_ignore,
        )

        assert len(res) == 1
        assert {0: 0, 1: 0, 2: 0} == res[0]
        box_match_single_mock.assert_called()

    # def test_box_matching_single_image_single_class(self):
    #     _ious = np.array([[0.4, 0.2, 0.2], [0.05, 0.4, 0.05], [0.1, 0.1, 0.9]])
    #     _iou_thresholds = np.array([0.1, 0.5])
    #     _dt_scores = np.array([0.9, 0.8, 0.7])
    #     _gt_ignore = np.array([0, 0, 1])
    #     res = _matching_single_image_single_class(_ious, _iou_thresholds, _dt_scores, _gt_ignore,
    #                                                  max_detections=100)
    #     self.assertTrue(np.isclose(res['dtMatches'], [[1., 1., 1.], [0., 0., 1.]]).all())
    #     self.assertTrue(np.isclose(res['gtMatches'], [[1., 1., 1.], [0., 0., 1.]]).all())
    #     self.assertTrue(np.isclose(res['dtScores'], [0.9, 0.8, 0.7]).all())
    #     self.assertTrue(np.isclose(res['gtIgnore'], [0, 0, 1]).all())
    #     self.assertTrue(np.isclose(res['dtIgnore'],  [[0., 0., 1.], [0., 0., 1.]]).all())
    #
    # def test_box_matching_single_image_single_class_mul_matches(self):
    #     _ious = np.array([[0.9, 0.2, 0.2], [0.8, 0.0, 0.1], [0.1, 0.1, 0.9]])
    #     _iou_thresholds = np.array([0.5])
    #     _dt_scores = np.array([0.9, 0.8, 0.7])
    #     _gt_ignore = np.array([0, 0, 1])
    #     res = _matching_single_image_single_class(_ious, _iou_thresholds, _dt_scores, _gt_ignore,
    #                                                  max_detections=100)
    #     self.assertTrue(np.isclose(res['dtMatches'], [[1., 0., 1.]]).all())
    #     self.assertTrue(np.isclose(res['gtMatches'], [[1., 0., 1.]]).all())

    def test_integration_box_matching_batch(self):
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
            box_iou_np,
            [0.1, 0.5, 0.75],
            _pd_boxes,
            _pd_classes,
            _pd_scores,
            _gt_boxes,
            _gt_classes,
            _gt_ignore,
        )
