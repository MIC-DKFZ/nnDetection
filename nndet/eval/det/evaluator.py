# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from abc import abstractclassmethod
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from loguru import logger

import nndet.core.ops_np as ops_np
from nndet.eval.abstract import AbstractEvalMatching, AbstractEvaluator, DetectionMetric
from nndet.eval.det.ap import CocoAPMetric
from nndet.eval.det.froc import FROCMetric, FROCwpMetric
from nndet.eval.det.hist import PredictionHistogram
from nndet.eval.matching import EvalMatchingPerElementGreedyScoreNP

__all__ = ["DetectionEvaluator"]


class DetectionEvaluator(AbstractEvaluator):
    def __init__(
        self,
        metrics: Sequence[DetectionMetric],
        matching: AbstractEvalMatching,
        criterion: Callable = ops_np.box_area_np,
        criterion_ranges: Optional[Dict[str, Tuple]] = None,
        save_dir: Optional[os.PathLike] = None,
    ):
        """
        Class for evaluate detection metrics

        Args:
            metrics: detection metrics to evaluate
            matching: object to perform matching for batches
            filter_keys: define keys which need to be filtered by the IoU value
            criterion: function that takes array of boxes [N, 4/6] and
                computes scalar criterion value array [N]
            criterion_ranges: Dict containing names and ranges of
                additional ranges of interest
            save_dir: if provided, this will call the plot function of the
                metric with the defined save_dir to create additional plots
        """
        self.criterion = criterion
        # set range to cover every object
        self.criterion_ranges = {"": (np.NINF, np.inf)}
        # expand by additional ranges
        if criterion_ranges is not None:
            for key, bounds in criterion_ranges.items():
                if key in self.criterion_ranges:
                    raise ValueError(f"Key {key} is not supported in criterion ranges since it is a default key")
                if bounds[1] < bounds[0]:
                    raise ValueError(
                        f"Bounds {bounds} from criterion ranges {criterion_ranges} are not supported."
                        "Upper bounds needs to larger than lower bound!"
                    )
            self.criterion_ranges.update(criterion_ranges)
        self.metrics = metrics
        self.matching = matching

        self.results_dict: Dict[str, Dict[Union[str, int], Dict]] = {
            key: OrderedDict() for key in self.criterion_ranges.keys()
        }  # store results of each image
        self.filter_keys = self.matching.get_filter_keys()
        self.iou_thresholds = self.get_unique_iou_thresholds()
        self.iou_mapping = self.get_indices_of_iou_for_each_metric()
        self.save_dir = Path(save_dir) if save_dir is not None else None

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
            pred_boxes: predicted boxes from single batch;
                List[[D, dim * 2]], D number of predictions
            pred_classes: predicted classes from a single batch; List[[D]],
                D number of predictions
            pred_scores: predicted score for each bounding box; List[[D]],
                D number of predictions
            gt_boxes: ground truth boxes; List[[G, dim * 2]], G number of
                ground truth
            gt_classes: ground truth classes; List[[G]], G number of ground
                truth
            gt_ignore: specified if which ground truth boxes are not counted
                as true positives (detections which match theses boxes are
                not counted as false positives either);
                List[[G]], G number of ground truth
            case_ids: optionally provide a case ids which will be return to
                identify the matching result. Integers are reserved for
                automatic counting.

        Returns
            dict: empty dict... detection metrics can only be evaluated
                at the end
        """
        # all criterion ranges should have the same number of images
        all_length = [len(v) for v in self.results_dict.values()]
        assert all([al == all_length[0] for al in all_length])

        if gt_ignore is None:
            n_gt = [0 if gt_boxes_img.size == 0 else gt_boxes_img.shape[0] for gt_boxes_img in gt_boxes]
            gt_ignore = [np.zeros(_n).reshape(-1) for _n in n_gt]

        # Compute criterion
        gt_boxes_criterion = [
            np.array([]) if gt_boxes_img.size == 0 else self.criterion(gt_boxes_img) for gt_boxes_img in gt_boxes
        ]
        dt_boxes_criterion = [
            np.array([]) if dt_boxes_img.size == 0 else self.criterion(dt_boxes_img) for dt_boxes_img in pred_boxes
        ]
        # Loop over all evaluated criterion ranges
        for results_key, criterion_range in self.criterion_ranges.items():
            criterion_pred_ignore, criterion_gt_ignore = self.get_criterion_ignores(
                criterion_range=criterion_range,
                gt_ignore=gt_ignore,
                gt_boxes_criterion=gt_boxes_criterion,
                dt_boxes_criterion=dt_boxes_criterion,
            )
            self.add_batch(
                results_key=results_key,
                pred_boxes=pred_boxes,
                pred_classes=pred_classes,
                pred_scores=pred_scores,
                pred_ignore=criterion_pred_ignore,
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
                gt_ignore=criterion_gt_ignore,
                case_ids=case_ids,
            )
        return {}

    @staticmethod
    def get_criterion_ignores(
        criterion_range: Tuple[int, int],
        gt_ignore: List[np.ndarray],
        gt_boxes_criterion: List[np.ndarray],
        dt_boxes_criterion: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Compute ignore arrays based on the provided criterion and criterion
        ranges

        Args:
            criterion_range: upper and lower bound of criterion
            gt_ignore: information on previously ignores ground truth objects
            gt_boxes_criterion: computed criterion information for ground truth
            dt_boxes_criterion: computed criterion information for predictions

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: tuple with two entries:
                first entry contains an [array] indicating new ignore values
                the predictions and second entry contain [array] for ground
                truth
        """
        # Define new gt_ignores based on the criterion
        gt_ignore_final = []
        for i, gt_boxes_img_criterion in enumerate(gt_boxes_criterion):
            gt_ignore_criterion = np.zeros(len(gt_ignore[i]), dtype=int)
            # If there is no ground truth in this image, we don't need to change the ignored values
            if not len(gt_ignore[i]) == 0:
                for j, gt_box_criterion in enumerate(gt_boxes_img_criterion):
                    if gt_box_criterion < criterion_range[0] or gt_box_criterion >= criterion_range[1]:
                        gt_ignore_criterion[j] = 1
            gt_ignore_final.append(np.logical_or(gt_ignore[i], gt_ignore_criterion))
        assert len(gt_ignore_final) == len(gt_ignore)

        # Find detections that are outside the criterion
        pred_outside = [
            np.logical_or(
                dt_box_criterion < criterion_range[0],
                dt_box_criterion >= criterion_range[1],
            )
            for dt_box_criterion in dt_boxes_criterion
        ]
        return pred_outside, gt_ignore_final

    def add_batch(
        self,
        results_key: str,
        pred_boxes: Sequence[np.ndarray],
        pred_classes: Sequence[np.ndarray],
        pred_scores: Sequence[np.ndarray],
        pred_ignore: Sequence[np.ndarray],
        gt_boxes: Sequence[np.ndarray],
        gt_classes: Sequence[np.ndarray],
        gt_ignore: Sequence[np.ndarray],
        case_ids: Optional[Sequence[str]] = None,
    ):
        """
        Match boxes of a batch to corresponding ground truth for each category
        independently

        Args:
            results_key: define key where batch should be added
            pred_boxes: predicted boxes from single batch; List[[D, dim * 2]],
                D number of predictions
            pred_classes: predicted classes from a single batch; List[[D]],
                D number of predictions
            pred_scores: predicted score for each bounding box; List[[D]],
                D number of predictions
            pred_ignore: boolean whether the predicted box should be ignored if
                it is not matched List[[D]]
            gt_boxes: ground truth boxes; List[[G, dim * 2]], G number of ground
                truth
            gt_classes: ground truth classes; List[[G]], G number of ground
                truth
            gt_ignore: specified if which ground truth boxes are not counted as
                true positives
                (detections which match theses boxes are not counted as false
                positives either); List[[G]], G number of ground truth
            case_ids: optionally provide case ids which will be returned to
                identify the matching result
        """
        batch_size = len(pred_boxes)
        if case_ids is None:
            # if no case ids are provided, we just count up
            n = len(self.results_dict[results_key])
            case_ids = list(range(n, n + batch_size))

        # check batch sizes
        if len(pred_classes) != batch_size:
            raise ValueError("Unequal batch size encountered for pred_classes.")
        if len(pred_scores) != batch_size:
            raise ValueError("Unequal batch size encountered for pred_scores.")
        if len(gt_boxes) != batch_size:
            raise ValueError("Unequal batch size encountered for gt_boxes.")
        if len(gt_classes) != batch_size:
            raise ValueError("Unequal batch size encountered for gt_classes.")
        if len(gt_ignore) != batch_size:
            raise ValueError("Unequal batch size encountered for gt_ignore.")
        if len(pred_ignore) != batch_size:
            raise ValueError("Unequal batch size encountered for pred_ignore.")
        if len(case_ids) != batch_size:
            raise ValueError("Unequal batch size encountered for case ids.")

        # iterate over images/batches
        for batch_idx, (pboxes, pclasses, pscores, pignore, gboxes, gclasses, gignore, case_id,) in enumerate(
            zip(
                pred_boxes,
                pred_classes,
                pred_scores,
                pred_ignore,
                gt_boxes,
                gt_classes,
                gt_ignore,
                case_ids,
            )
        ):
            self.results_dict[results_key][case_id] = self.matching.match(
                iou_thresholds=self.iou_thresholds,
                pred_boxes=pboxes,
                pred_classes=pclasses,
                pred_scores=pscores,
                pred_ignore=pignore,
                gt_boxes=gboxes,
                gt_classes=gclasses,
                gt_ignore=gignore,
            )

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
                iou_filtered_results = [
                    self.iou_filter(
                        r,
                        iou_idx=self.iou_mapping[metric_idx],
                        filter_keys=self.filter_keys,
                    )
                    for r in results.values()
                ]

                _criterion_key = criterion_key if criterion_key else None
                score, curve = metric(iou_filtered_results, tag=_criterion_key)
                if self.save_dir is not None:
                    metric.plot(score, curve, save_dir=self.save_dir, tag=_criterion_key)

                if score is not None:
                    metric_scores.update(score)

                if curve is not None:
                    metric_curves.update(curve)
        # Add entries containing the criterion ranges
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
            filter_keys: keys to filter, defined by matching

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
        self.results_dict = {key: OrderedDict() for key in self.criterion_ranges.keys()}

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
        criterion: Callable = ops_np.box_area_np,
        criterion_ranges: Optional[Dict[str, Tuple]] = None,
        froc_wp: bool = True,
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
            criterion: Criterion for separate evaluation
            criterion_ranges: Ranges of the value of the box criterion to
                evaluate (the first entry should be "": full range

        Returns:
            BoxEvaluator: evaluator to efficiently compute metrics
        """
        max_detections = os.getenv("nndet_eval_max_detections_image", 100)
        iou_range = (0.1, 0.5, 0.05)
        iou_thresholds = (0.1, 0.5) if fast else (0.1, 0.2, 0.3, 0.5)
        criterion_ranges_final = {
            # non overlapping default set
            "sVF": (0, 8**3),
            "mVF": (8**3, 24**3),
            "lVF": (24**3, np.inf),
            # extended analysis
            "xxsVF": (0, 4**3),
            "xsVF": (0, 6**3),
            "xlVF": (32**3, np.inf),
            "xxlVF": (48**3, np.inf),
            "xxxlVF": (64**3, np.inf),
        }
        if criterion_ranges is not None and not fast:
            criterion_ranges_final.update(criterion_ranges)

        metrics = []
        froc_cls = FROCwpMetric if froc_wp else FROCMetric
        metrics.append(
            froc_cls(
                classes,
                iou_thresholds=iou_thresholds,
                fpi_thresholds=(1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8),
                verbose=verbose,
            )
        )
        metrics.append(
            CocoAPMetric(
                classes,
                iou_list=iou_thresholds,
                iou_range=iou_range,
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
        logger.info(
            f"Created {cls.__name__} (box vol/area criterion) with "
            f"{matching.__class__.__name__} and {max_detections} max detections."
        )
        return cls(
            metrics=tuple(metrics),
            matching=matching,
            criterion=criterion,
            criterion_ranges=criterion_ranges_final,
            save_dir=save_dir,
        )
