# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0


from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import nndet.core.ops_np as ops_np
from nndet.evaluator.abstract import AbstractEvaluator, DetectionMetric
from nndet.evaluator.detection.coco import COCOMetric
from nndet.evaluator.detection.froc import FROCMetric
from nndet.evaluator.detection.hist import PredictionHistogram
from nndet.evaluator.detection.matching import matching_batch
from nndet.utils.info import experimental

__all__ = ["DetectionEvaluator"]


class DetectionEvaluator(AbstractEvaluator):
    def __init__(
        self,
        metrics: Sequence[DetectionMetric],
        iou_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        max_detections: int = 100,
        match_fn: Callable = matching_batch,
        filter_keys: Sequence[str] = ("dtMatches", "gtMatches", "dtIgnore"),
        box_criterion: Callable = ops_np.box_area_np,
        criterion_ranges: Optional[Dict[str, Tuple]] = None,
    ):
        """
        Class for evaluate detection metrics

        Args:
            metrics: detection metrics to evaluate
            iou_fn: compute overlap for each pair
            max_detections: number of maximum detections per image
                (reduces computation)
            match_fn: function to match predictions to ground truth
            filter_keys: define keys which need to be filtered by the IoU value
            box_criterion: function that takes array of boxes [N, 4/6] and computes scalar criterion value
                array [N]
            criterion_ranges: (optional) Dict containing names and ranges of additional ranges of interest
        """
        self.iou_fn = iou_fn
        self.match_fn = match_fn

        self.max_detections = max_detections
        self.box_criterion = box_criterion
        # set range to cover every object
        self.criterion_ranges = {"": (np.NINF, np.inf)}
        # expand by additional ranges
        if criterion_ranges is not None:
            self.criterion_ranges.update(criterion_ranges)

        self.metrics = metrics
        self.filter_keys = filter_keys

        self.results_dict = {key: [] for key in self.criterion_ranges.keys()}  # store results of each image

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
        for results_key, criterion_range in self.criterion_ranges.items():
            # Define new gt_ignores based on the criterion
            gt_ignore_final = []
            for i, gt_boxes_img_criterion in enumerate(gt_boxes_criterion):
                # TODO maybe should switch to use explicit boolean and not zeros?
                gt_ignore_criterion = np.zeros(gt_ignore[i].shape)
                # If there is no ground truth in this image, we don't need to change the ignored values
                if not gt_ignore[i].size == 0:
                    gt_ignore_criterion = (gt_boxes_img_criterion <= criterion_range[0]) | (
                        gt_boxes_img_criterion > criterion_range[1]
                    )
                gt_ignore_final.append(np.logical_or(gt_ignore[i], gt_ignore_criterion))
            # Get all matches
            temp_matches = self.match_fn(
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
            )
            # Find unmatched detections and ignore those not fitting to criterion
            self.results_dict[results_key].extend(
                self.find_dt_ignores(
                    results_key,
                    temp_matches,
                    self.iou_thresholds,
                    pred_boxes=pred_boxes,
                    pred_classes=pred_classes,
                    pred_scores=pred_scores,
                    gt_boxes=gt_boxes,
                    gt_classes=gt_classes,
                    gt_ignore=gt_ignore_final,
                    max_detections=self.max_detections,
                )
            )
        return {}

    def find_dt_ignores(
        self,
        results_key: str,
        matches: List[Dict[int, Dict[str, np.ndarray]]],
        iou_thresholds: Sequence[float],
        pred_boxes: Sequence[np.ndarray],
        pred_classes: Sequence[np.ndarray],
        pred_scores: Sequence[np.ndarray],
        gt_boxes: Sequence[np.ndarray],
        gt_classes: Sequence[np.ndarray],
        gt_ignore: Sequence[Sequence[bool]],
        max_detections: int = 100,
    ):
        # iterate over images/batches
        for match, pboxes, pclasses, pscores, gboxes, gclasses, gignore in zip(
            matches, pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes, gt_ignore
        ):
            img_classes = np.union1d(pclasses, gclasses)
            for c in img_classes:
                pred_mask = pclasses == c  # mask predictions with current class
                if not np.any(pred_mask):  # no predictions
                    continue
                # if there are predictions, find unmatched predictions outside the ranges and add to dtIgnore
                pred_boxes_masked = pboxes[pred_mask]
                pred_scores_masked = pscores[pred_mask]
                # filter for max_detections highest scoring predictions to speed up computation
                dt_ind = np.argsort(-pred_scores_masked, kind="mergesort")
                dt_ind = dt_ind[:max_detections]

                pred_boxes_sorted = pred_boxes_masked[dt_ind]
                dt_match = match[c]["dtMatches"]
                dt_ignore = match[c]["dtIgnore"]
                # Calculate the box criterion for all boxes
                dt_boxes_criterion = self.box_criterion(pred_boxes_sorted)
                # Find outliers
                dt_outside = np.array(
                    [
                        dt_box_criterion <= self.criterion_ranges[results_key][0]
                        or dt_box_criterion > self.criterion_ranges[results_key][1]
                        for dt_box_criterion in dt_boxes_criterion
                    ]
                ).reshape(1, -1)
                # ignore outliers and previous ignores
                dt_ignore = np.logical_or(
                    dt_ignore, np.logical_and(dt_match == 0, np.repeat(dt_outside, len(iou_thresholds), axis=0))
                )
                match[c]["dtIgnore"] = dt_ignore
        return matches

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
            for criterion_key, results in self.results_dict.items():
                _filter = partial(
                    self.iou_filter,
                    iou_idx=self.iou_mapping[metric_idx],
                    filter_keys=self.filter_keys,
                )
                iou_filtered_results = list(map(_filter, results))
                score, curve = metric(iou_filtered_results, tag=criterion_key)

                if score is not None:
                    metric_scores.update(score)

                if curve is not None:
                    metric_curves.update(curve)
        metric_curves.update(
            {
                f"criterion_{tag}" if tag != "" else "criterion": criterion_range
                for tag, criterion_range in self.criterion_ranges.items()
            }
        )
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
        self.results_dict = {key: [] for key in self.criterion_ranges.keys()}

    @classmethod
    def create(
        cls,
        classes: Sequence[str],
        fast: bool = True,
        verbose: bool = False,
        save_dir: Optional[Path] = None,
        box_criterion: Callable = ops_np.box_area_np,
        criterion_ranges: Optional[Dict[str, Tuple]] = None,
    ):
        """
        Create a box evaluator object

        Args:
            classes: classes present in the dataset
            fast: Reduces the evaluation suite to save time.
                Only evaluated IoUs in the range of 0.1-0.5
                Does not calculate pre-class metrics
            verbose: Additional logging output
            save_dir: Path to save information
            box_criterion: Criterion for separate evaluation
            criterion_ranges: Ranges of the value of the box criterion to evaluate (the first entry should be
                "": full range

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

        return cls(
            metrics=tuple(metrics),
            iou_fn=cls.similarity_fn,
            box_criterion=box_criterion,
            criterion_ranges=criterion_ranges,
        )


class BoxEvaluator(DetectionEvaluator):
    similarity_fn = ops_np.box_iou_np


class MaskEvaluator(DetectionEvaluator):
    similarity_fn = ops_np.bin_mask_iou_np


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
