# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractclassmethod
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import nndet.core.ops_np as ops_np
from nndet.eval.abstract import AbstractEvalMatching, AbstractEvaluator, DetectionMetric
from nndet.eval.detection.coco import COCOMetric
from nndet.eval.detection.froc import FROCMetric
from nndet.eval.detection.hist import PredictionHistogram
from nndet.eval.matching import EvalMatchingPerElementGreedyScoreNP
from nndet.utils.info import experimental

__all__ = ["DetectionEvaluator"]


class DetectionEvaluator(AbstractEvaluator):
    def __init__(
        self,
        metrics: Sequence[DetectionMetric],
        matching: AbstractEvalMatching,
    ):
        """
        Class for evaluate detection metrics

        Args:
            metrics: detection metrics to evaluate
            matching: object to perform matching for batches
            filter_keys: define keys which need to be filtered by the IoU value
        """
        self.metrics = metrics
        self.matching = matching

        self.results_list = []  # store results of each image
        self.filter_keys = self.matching.get_filter_keys()
        self.iou_thresholds = self.get_unique_iou_thresholds()
        self.iou_mapping = self.get_indices_of_iou_for_each_metric()

    def get_unique_iou_thresholds(self):
        """
        Compute unique set of iou thresholds
        """
        iou_thresholds = [_i for i in self.metrics for _i in i.get_iou_thresholds()]
        iou_thresholds = list(set(iou_thresholds))
        iou_thresholds.sort()
        return iou_thresholds

    def get_indices_of_iou_for_each_metric(self):
        """
        Find indices of iou thresholds for each metric
        """
        return [[self.iou_thresholds.index(th) for th in m.get_iou_thresholds()] for m in self.metrics]

    def run_online_evaluation(
        self,
        pred_boxes: Sequence[np.ndarray],
        pred_classes: Sequence[np.ndarray],
        pred_scores: Sequence[np.ndarray],
        gt_boxes: Sequence[np.ndarray],
        gt_classes: Sequence[np.ndarray],
        gt_ignore: Sequence[Sequence[bool]] = None,
        case_ids: Optional[Sequence[str]] = None,
    ) -> Dict:
        """
        Preprocess batch results for final evaluation

        Args:
            pred_boxes (Sequence[np.ndarray]): predicted boxes from single batch; List[[D, dim * 2]], D number of
                predictions
            pred_classes (Sequence[np.ndarray]): predicted classes from a single batch; List[[D]], D number of
                predictions
            pred_scores (Sequence[np.ndarray]): predicted score for each bounding box; List[[D]], D number of
                predictions
            gt_boxes (Sequence[np.ndarray]): ground truth boxes; List[[G, dim * 2]], G number of ground truth
            gt_classes (Sequence[np.ndarray]): ground truth classes; List[[G]], G number of ground truth
            gt_ignore (Sequence[Sequence[bool]]): specified if which ground truth boxes are not counted as true
                positives (detections which match theses boxes are not counted as false positives either);
                List[[G]], G number of ground truth
            case_ids: optionally provide a case ids which will be return to
                identify the matching result

        Returns
            dict: empty dict... detection metrics can only be evaluated at the end
        """
        if gt_ignore is None:
            n = [0 if gt_boxes_img.size == 0 else gt_boxes_img.shape[0] for gt_boxes_img in gt_boxes]
            gt_ignore = [np.zeros(_n).reshape(-1) for _n in n]

        self.results_list.extend(
            self.matching.match(
                iou_thresholds=self.iou_thresholds,
                pred_boxes=pred_boxes,
                pred_classes=pred_classes,
                pred_scores=pred_scores,
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
                gt_ignore=gt_ignore,
                case_ids=case_ids,
            )
        )
        return {}

    def finish_online_evaluation(
        self,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Accumulate results of individual batches and compute final metrics

        Returns:
            Dict[str, float]: dictionary with scalar values for evaluation
            Dict[str, np.ndarray]: dictionary with arrays, e.g. for visualization of graphs
        """
        metric_scores = {}
        metric_curves = {}
        for metric_idx, metric in enumerate(self.metrics):
            _filter = partial(
                self.iou_filter,
                iou_idx=self.iou_mapping[metric_idx],
                filter_keys=self.filter_keys,
            )
            iou_filtered_results = list(map(_filter, self.results_list))

            score, curve = metric(iou_filtered_results)

            if score is not None:
                metric_scores.update(score)

            if curve is not None:
                metric_curves.update(curve)
        return metric_scores, metric_curves

    @staticmethod
    def iou_filter(
        image_dict: Dict[int, Dict[str, np.ndarray]],
        iou_idx: List[int],
        filter_keys: Sequence[str],
    ):
        """
        This functions can be used to filter specific IoU values from the results
        to make sure that the correct IoUs are passed to metric

        Args:
            image_dict: dictionary containin :param:`filter_keys`
                which contains IoUs in the first dimension
            iou_idx: indices of IoU values to filter from keys
            filter_keys: keys to filter, by default
                ('dtMatches', 'gtMatches', 'dtIgnore')

        Returns
            dict: filtered dictionary
        """
        iou_idx = list(iou_idx)
        filtered = {}
        for cls_key, cls_item in image_dict.items():
            filtered[cls_key] = {key: item[iou_idx] if key in filter_keys else item for key, item in cls_item.items()}
        return filtered

    def reset(self):
        """
        Reset internal state of evaluator
        """
        self.results_list = []

    @abstractclassmethod
    def create(
        cls,
        classes: Sequence[str],
        fast: bool = True,
        verbose: bool = False,
        save_dir: Optional[Path] = None,
    ) -> DetectionEvaluator:
        """
        Create an evaluator object

        Args:
            classes: classes present in the dataset
            fast: Reduces the evaluation suite to save time (e.g. during
                training)
            verbose: Additional logging output
            save_dir: Path to save information

        Returns:
            DetectionEvaluator: evaluator to efficiently compute metrics
        """
        raise NotImplementedError


class BoxEvaluator(DetectionEvaluator):
    similarity_fn = ops_np.box_iou_np

    @classmethod
    def create(
        cls,
        classes: Sequence[str],
        fast: bool = True,
        verbose: bool = False,
        save_dir: Optional[Path] = None,
    ) -> BoxEvaluator:
        """
        Create an box evaluator object

        Args:
            classes: classes present in the dataset
            fast: Reduces the evaluation suite to save time.
                Only evaluated IoUs in the range of 0.1-0.5
                Does no calculate pre class metrics
            verbose: Additional logging output
            save_dir: Path to save information

        Returns:
            BoxEvaluator: evaluator to efficiently compute metrics
        """
        max_detections = 100
        iou_range = (0.1, 0.5, 0.05)
        iou_thresholds = (0.1, 0.5) if fast else np.arange(0.1, 1.0, 0.1)

        metrics = []
        metrics.append(
            FROCMetric(
                classes,
                iou_thresholds=iou_thresholds,
                fpi_thresholds=(1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8),
                verbose=verbose,
                save_dir=None if fast else save_dir,
            )
        )
        metrics.append(
            COCOMetric(
                classes,
                iou_list=iou_thresholds,
                iou_range=iou_range,
                max_detection=(100,),
                verbose=verbose,
            )
        )

        if not fast:
            metrics.append(
                PredictionHistogram(
                    classes=classes,
                    save_dir=save_dir,
                    iou_thresholds=(0.1, 0.5),
                )
            )

        matching = EvalMatchingPerElementGreedyScoreNP(
            iou_fn=cls.similarity_fn,
            max_detections=max_detections,
            warning_ratio=0.25,
        )
        return cls(
            metrics=tuple(metrics),
            matching=matching,
        )


"""
############ Experimental Evaluators ############
"""


class CountDifferenceEvaluator(AbstractEvaluator):
    @experimental
    def __init__(self, min_prob: float = 0.5):
        super().__init__()
        self.min_prob = min_prob

        self.num_gt = []
        self.num_pred = []

    def run_online_evaluation(
        self,
        pred_scores: Sequence[np.ndarray],
        gt_classes: Sequence[np.ndarray],
    ) -> Dict:
        """
        Preprocess batch results for final evaluation

        Args:
            pred_scores: predicted score for each bounding box; List[[D]],
                D number of predictions
            gt_classes: ground truth classes; List[[G]], G number of ground
                truth

        Returns
            dict: empty dict
        """
        assert len(pred_scores) == len(gt_classes)
        for p, g in zip(pred_scores, gt_classes):
            if p.size > 0:
                self.num_pred.append((p > self.min_prob).sum())
            else:
                self.num_pred.append(0)
            self.num_gt.append(len(g))
        return {}

    def finish_online_evaluation(
        self,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Accumulate results of individual batches and compute final metrics

        Returns:
            Dict[str, float]: dictionary with scalar values for evaluation
                `mean`: mean number of count differences
                `median`: median number of count differences
                `max`: max number of count differences
                `min`: min number of count differences
            Dict[str, np.ndarray]: absolute difference per case
                `diff_per_case`: count difference per case
                `diff_per_case_sign`: count difference per case signed
                    computed as: #gt - #pred
        """
        gts = np.asarray(self.num_gt)
        preds = np.asarray(self.num_pred)

        diff_per_case = gts - preds
        metric_scores = {
            "mean": np.mean(np.absolute(diff_per_case)),
            "median": np.median(np.absolute(diff_per_case)),
            "max": np.max(np.absolute(diff_per_case)),
            "min": np.min(np.absolute(diff_per_case)),
        }
        metric_curves = {
            "diff_per_case": np.absolute(diff_per_case),
            "diff_per_case_sign": diff_per_case,
        }
        return metric_scores, metric_curves

    def reset(self):
        """
        Reset internal state of evaluator
        """
        self.num_gt = []
        self.num_pred = []
