# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import os
from abc import ABC, abstractclassmethod, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["AbstractEvaluator", "AbstractMetric", "DetectionMetric"]


class AbstractEvaluator(ABC):
    @abstractmethod
    def run_online_evaluation(self, *args, **kwargs):
        """
        Compute necessary values per batch for later evaluation
        """
        raise NotImplementedError

    @abstractmethod
    def finish_online_evaluation(self, *args, **kwargs):
        """
        Accumulate results from batches and compute metrics
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """
        Reset internal state of evaluator
        """
        raise NotImplementedError


class AbstractMetric(ABC):
    def __call__(
        self,
        *args,
        **kwargs,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Compute metric. See :func:`compute` for more information.

        Args:
            *args: positional arguments passed to :func:`compute`
            **kwargs: keyword arguments passed to :func:`compute`

        Returns:
            Dict[str, float]: dictionary with scalar values for evaluation
            Dict[str, np.ndarray]: dictionary with arrays, e.g. for
                visualization of graphs
        """
        return self.compute(*args, **kwargs)

    @abstractmethod
    def compute(
        self,
        results_list: List[Dict[int, Dict[str, np.ndarray]]],
        tag: Optional[str] = "",
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Compute metric

        Args:
            results_list: list with result s per image (in list) per category
                (dict). Inner Dict contains multiple results obtained
                by :func:`box_matching_batch`.

                ``dtMatches``: np.ndarray
                    matched detections [T, D], where T = number of thresholds,
                    D = number of detections

                ``gtMatches``: np.ndarray
                    matched ground truth boxes [T, G], where T = number of
                    thresholds, G = number of ground truth

                ``dtScores``: np.ndarray
                    prediction scores [D] detection scores

                ``gtIgnore``: np.ndarray
                    ground truth boxes which should be ignored [G] indicate
                    whether ground truth should be ignored

                ``dtIgnore``: np.ndarray
                    detections which should be ignored [T, D], indicate
                    which detections should be ignored

            tag: tag of the current evaluation. Added to metric keys and
                filenames. If None, no tag will be used

        Returns:
            Dict[str, float]: dictionary with scalar values for evaluation
            Dict[str, Any]: Contains additional meta data e.g. underlying
                curves or debug information.
        """
        raise NotImplementedError

    @classmethod
    def plot(
        cls,
        result_scores: Dict[str, float],
        result_curves: Dict[str, Any],
        save_dir: Optional[os.PathLike] = None,
    ) -> Dict:
        """
        Plot curves which might have been generated during the evaluation

        Args:
            result_scores: single scores from metric
            result_curves: meta data
            save_dir: path to directory where files should be saved. If None,
                the plots won't be saved

        Returns:
            Dict: figures of create plots
        """
        pass


class DetectionMetric(AbstractMetric):
    @staticmethod
    def get_name(tag: Optional[str] = None) -> str:
        """
        Return name of file to save

        Returns:
            str: Name of the Metric and the chosen setting
        """
        raise NotImplementedError

    @abstractmethod
    def get_iou_thresholds(self) -> Sequence[float]:
        """
        Return IoU thresholds needed for this metric in an numpy array

        Returns:
            Sequence[float]: IoU thresholds; [M], M is the number of thresholds
        """
        raise NotImplementedError

    def check_number_of_iou(self, *args) -> None:
        """
        Check if shape of input in first dimension is consistent with expected IoU values
        (assumes IoU dimension is the first dimension)

        Args:
            args: array like inputs with shape function
        """
        num_ious = len(self.get_iou_thresholds())
        for arg in args:
            assert arg.shape[0] == num_ious


class AbstractEvalMatching(ABC):
    def __init__(
        self,
        iou_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        max_detections: int,
        warning_ratio: float = 0.25,
    ) -> None:
        """
        Perform matching of predictions and ground truth for evaluation.

        Args:
            iou_fn: compute overlap for each pair
            max_detections: maximum number of detections which should be
                evaluated (per class)
            warning_ratio: if number of ground truth exceeds
                `max_detections * warning_ratio` during evaluation a warning
                is shown
        """
        super().__init__()
        self.iou_fn = iou_fn
        self.max_detections = max_detections
        self.warning_ratio = warning_ratio

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(iou_fn: {self.iou_fn.__name__}, max_detections: {self.max_detections}, "
            f"warning_ratio: {self.warning_ratio})"
        )

    @abstractclassmethod
    def get_filter_keys(cls) -> List[str]:
        """
        Return keys which need to be filtered by IoU values

        Returns:
            List[str]: name of keys which need to be filtered
        """
        raise NotImplementedError

    @abstractmethod
    def match(
        self,
        iou_thresholds: Sequence[float],
        pred_boxes: Sequence[np.ndarray],
        pred_classes: Sequence[np.ndarray],
        pred_scores: Sequence[np.ndarray],
        gt_boxes: Sequence[np.ndarray],
        gt_classes: Sequence[np.ndarray],
        gt_ignore: Sequence[Sequence[bool]],
        case_id: Optional[str] = None,
    ) -> List[Dict[int, Dict[str, np.ndarray]]]:
        """
        Match boxes of a batch to corresponding ground truth for each category
        independently

        Args:
            iou_thresholds: defined which IoU thresholds should be evaluated
            pred_boxes: predicted boxes from single batch; List[[D, dim * 2]],
                D number of predictions
            pred_classes: predicted classes from a single batch; List[[D]],
                D number of predictions
            pred_scores: predicted score for each bounding box; List[[D]],
                D number of predictions
            gt_boxes: ground truth boxes; List[[G, dim * 2]], G number of ground
                truth
            gt_classes: ground truth classes; List[[G]], G number of ground truth
            gt_ignore: specified if which ground truth boxes are not counted as
                true positives
                (detections which match theses boxes are not counted as false
                positives either); List[[G]], G number of ground truth
            case_id: optionally provide a case id which will be returned to
                identify the matching result

        Returns:
            List[Dict[int, Dict[str, np.ndarray]]]
                matched detections [dtMatches] and ground truth [gtMatches]
                boxes [str, np.ndarray] for each category (stored in dict keys)
                for each image (list)
        """
        raise NotImplementedError
