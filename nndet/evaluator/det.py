"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pathlib import Path
from functools import partial
from typing import Optional, Sequence, Callable, Dict, List, Tuple

import numpy as np

from nndet.evaluator.abstract import AbstractEvaluator, DetectionMetric
from nndet.evaluator.detection.matching import matching_batch
from nndet.core.boxes import box_iou_np
from nndet.evaluator.detection.coco import COCOMetric
from nndet.evaluator.detection.froc import FROCMetric
from nndet.evaluator.detection.hist import PredictionHistogram


__all__ = ["DetectionEvaluator"]


class DetectionEvaluator(AbstractEvaluator):
    def __init__(self,
                 metrics: Sequence[DetectionMetric],
                 iou_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = box_iou_np,
                 match_fn: Callable = matching_batch,
                 max_detections: int = 100,
                 ):
        """
        Class for evaluate detection metrics

        Args:
            metrics (Sequence[DetectionMetric]: detection metrics to evaluate
            iou_fn (Callable[[np.ndarray, np.ndarray], np.ndarray]): compute overlap for each pair
            max_detections (int): number of maximum detections per image (reduces computation)
        """
        self.iou_fn = iou_fn
        self.match_fn = match_fn
        self.max_detections = max_detections
        self.metrics = metrics
        self.results_list = []  # store results of each image

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
        return [[self.iou_thresholds.index(th) for th in m.get_iou_thresholds()]
                for m in self.metrics]

    def run_online_evaluation(self,
                              pred_boxes: Sequence[np.ndarray],
                              pred_classes: Sequence[np.ndarray],
                              pred_scores: Sequence[np.ndarray],
                              gt_boxes: Sequence[np.ndarray],
                              gt_classes: Sequence[np.ndarray],
                              gt_ignore: Sequence[Sequence[bool]] = None) -> Dict:
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

        Returns
            dict: empty dict... detection metrics can only be evaluated at the end
        """
        if gt_ignore is None:
            n = [0 if gt_boxes_img.size == 0 else gt_boxes_img.shape[0] for gt_boxes_img in gt_boxes]
            gt_ignore = [np.zeros(_n).reshape(-1) for _n in n]

        self.results_list.extend(self.match_fn(
            self.iou_fn, self.iou_thresholds, pred_boxes=pred_boxes, pred_classes=pred_classes,
            pred_scores=pred_scores, gt_boxes=gt_boxes, gt_classes=gt_classes, gt_ignore=gt_ignore,
            max_detections=self.max_detections))

        return {}

    def finish_online_evaluation(self) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Accumulate results of individual batches and compute final metrics

        Returns:
            Dict[str, float]: dictionary with scalar values for evaluation
            Dict[str, np.ndarray]: dictionary with arrays, e.g. for visualization of graphs
        """
        metric_scores = {}
        metric_curves = {}
        for metric_idx, metric in enumerate(self.metrics):
            _filter = partial(self.iou_filter, iou_idx=self.iou_mapping[metric_idx])
            iou_filtered_results = list(map(_filter, self.results_list))
            
            score, curve = metric(iou_filtered_results)
            
            if score is not None:
                metric_scores.update(score)
            
            if curve is not None:
                metric_curves.update(curve)
        return metric_scores, metric_curves

    @staticmethod
    def iou_filter(image_dict: Dict[int, Dict[str, np.ndarray]], iou_idx: List[int],
                   filter_keys: Sequence[str] = ('dtMatches', 'gtMatches', 'dtIgnore')):
        """
        This functions can be used to filter specific IoU values from the results
        to make sure that the correct IoUs are passed to metric
        
        Parameters
        ----------
        image_dict : dict
            dictionary containin :param:`filter_keys` which contains IoUs in the first dimension
        iou_idx : List[int]
            indices of IoU values to filter from keys
        filter_keys : tuple, optional
            keys to filter, by default ('dtMatches', 'gtMatches', 'dtIgnore')
        
        Returns
        -------
        dict
            filtered dictionary
        """
        iou_idx = list(iou_idx)
        filtered = {}
        for cls_key, cls_item in image_dict.items():
            filtered[cls_key] = {key: item[iou_idx] if key in filter_keys else item
                                 for key, item in cls_item.items()}
        return filtered

    def reset(self):
        """
        Reset internal state of evaluator
        """
        self.results_list = []


class BoxEvaluator(DetectionEvaluator):
    @classmethod
    def create(cls,
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
        iou_fn = box_iou_np
        iou_range = (0.1, 0.5, 0.05)
        iou_thresholds = (0.1, 0.5) if fast else np.arange(0.1, 1.0, 0.1)
        per_class = False if fast else True

        metrics = []
        metrics.append(
            FROCMetric(classes,
                       iou_thresholds=iou_thresholds,
                       fpi_thresholds=(1/8, 1/4, 1/2, 1, 2, 4, 8),
                       per_class=per_class,
                       verbose=verbose,
                       save_dir= None if fast else save_dir
                       )
            )
        metrics.append(
            COCOMetric(classes,
                       iou_list=iou_thresholds,
                       iou_range=iou_range,
                       max_detection=(100, ),
                       per_class=per_class,
                       verbose=verbose,
                       )
            )

        if not fast:
            metrics.append(
                PredictionHistogram(classes=classes,
                                    save_dir=save_dir,
                                    iou_thresholds=(0.1, 0.5),
                                    )
                )
        return cls(metrics=tuple(metrics), iou_fn=iou_fn)
