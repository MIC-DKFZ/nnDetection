# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0


import copy

# Avoid have OrderedDict twice
from collections import OrderedDict as ODict
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, OrderedDict, Sequence, Tuple

import numpy as np

from nndet.core.boxes import box_area_np, box_iou_np
from nndet.core.masks.ops_np import bin_mask_iou_np
from nndet.evaluator.abstract import AbstractEvaluator, DetectionMetric
from nndet.evaluator.detection.coco import COCOMetric
from nndet.evaluator.detection.froc import FROCMetric
from nndet.evaluator.detection.hist import PredictionHistogram
from nndet.evaluator.detection.matching import matching_batch
from nndet.utils.info import experimental

__all__ = ["DetectionEvaluator"]


class DetectionEvaluator(AbstractEvaluator):
    similarity_fn = box_iou_np

    def __init__(
        self,
        metrics: Sequence[DetectionMetric],
        iou_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = box_iou_np,
        max_detections: int = 100,
        match_fn: Callable = matching_batch,
        box_criterion: Callable = box_area_np,
        criterion_ranges: OrderedDict[str, Tuple] = None,
        filter_keys: Sequence[str] = ("dtMatches", "gtMatches", "dtIgnore"),
    ):
        """
        Class for evaluate detection metrics

        Args:
            metrics: detection metrics to evaluate
            iou_fn: compute overlap for each pair
            max_detections: number of maximum detections per image
                (reduces computation)
            match_fn: function to match predictions to ground truth
            box_criterion: function that takes np.array of boxes [N, 4/6] and computes scalar criterion value
                np.array [N]
            criterion_ranges: OrderedDict containing names and ranges of the different ranges of interest, the first
                entry should contain the whole region otherwise the evaluation is not complete
            filter_keys: define keys which need to be filtered by the IoU value
        """
        self.iou_fn = iou_fn
        self.match_fn = match_fn

        self.max_detections = max_detections
        self.box_criterion = box_criterion
        # Set default ranges here to not have mutable parameter
        if criterion_ranges is None:
            criterion_ranges = ODict(
                {"": (0, 128**3), "_S": (0, 10**3), "_M": (10**3, 24**3), "_L": (24**3, 128**3)}
            )
        self.criterion_ranges = criterion_ranges
        self.metrics = metrics
        self.filter_keys = filter_keys

        self.results_list = [[] for i in criterion_ranges]  # store results of each image

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
        case_id: Optional[str] = None,
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
            case_id: optionally provide a case id which will be return to
                identify the matching result

        Returns
            dict: empty dict... detection metrics can only be evaluated at the end
        """
        if gt_ignore is None:
            n = [0 if gt_boxes_img.size == 0 else gt_boxes_img.shape[0] for gt_boxes_img in gt_boxes]
            gt_ignore = [np.zeros(_n).reshape(-1) for _n in n]

        # Compute ground truth volumes, set volume of no gt to -1 to ignore
        gt_boxes_criterion = [
            np.array([-1]) if gt_boxes_img.size == 0 else self.box_criterion(gt_boxes_img) for gt_boxes_img in gt_boxes
        ]

        # Loop over all evaluated criterion ranges
        for list_index, criterion_range in enumerate(self.criterion_ranges.values()):
            # Define new gt_ignores based on the criterion
            gt_ignore_final = copy.deepcopy(gt_ignore)
            for i, gt_boxes_img_criterion in enumerate(gt_boxes_criterion):
                # If there is no ground truth in this image, we don't need to change the ignored values
                if not gt_ignore_final[i].size == 0:
                    for j, gt_box_criterion in enumerate(gt_boxes_img_criterion):
                        if (
                            gt_ignore_final[i][j]
                            or gt_box_criterion < criterion_range[0]
                            or gt_box_criterion >= criterion_range[1]
                        ):
                            gt_ignore_final[i][j] = 1
                        else:
                            gt_ignore_final[i][j] = 0
            self.results_list[list_index].extend(
                self.match_fn(
                    self.iou_fn,
                    self.iou_thresholds,
                    pred_boxes=pred_boxes,
                    pred_classes=pred_classes,
                    pred_scores=pred_scores,
                    gt_boxes=gt_boxes,
                    gt_classes=gt_classes,
                    gt_ignore=gt_ignore_final,
                    max_detections=self.max_detections,
                    case_id=case_id,
                    criterion=self.box_criterion,
                    criterion_range=criterion_range,
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
            for criterion_key, results in zip(self.criterion_ranges.keys(), self.results_list):
                _filter = partial(
                    self.iou_filter,
                    iou_idx=self.iou_mapping[metric_idx],
                    filter_keys=self.filter_keys,
                )
                iou_filtered_results = list(map(_filter, results))

                if metric.__class__ != COCOMetric:
                    score, curve = metric(iou_filtered_results, title_prefix=criterion_key)
                else:
                    score, curve = metric(iou_filtered_results)
                if score is not None:
                    score = {f"{key}{criterion_key}": value for key, value in score.items()}
                    metric_scores.update(score)

                if curve is not None:
                    curve = {f"{key}_{criterion_key}": value for key, value in curve.items()}
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

    @classmethod
    def create(
        cls,
        classes: Sequence[str],
        fast: bool = True,
        verbose: bool = False,
        save_dir: Optional[Path] = None,
    ):
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
        # iou_fn = box_iou_np
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
        return cls(metrics=tuple(metrics), iou_fn=cls.similarity_fn)


class BoxEvaluator(DetectionEvaluator):
    similarity_fn = box_iou_np


class MaskEvaluator(DetectionEvaluator):
    similarity_fn = bin_mask_iou_np


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
