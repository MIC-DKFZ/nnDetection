# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    def get_tags(tag: Optional[str] = None) -> str:
        """
        Return name of file to save

        Returns:
            str: Name of the Metric and the chosen setting
            str: Tag Prefix for meta information
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
