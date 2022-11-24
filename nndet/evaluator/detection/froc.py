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

import time
import numpy as np

from loguru import logger
from typing import Sequence, List, Dict, Optional, Union, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from nndet.evaluator import DetectionMetric
from sklearn.metrics import roc_curve
from collections import defaultdict

from nndet.utils.info import experimental


class FROCMetric(DetectionMetric):
    def __init__(self,
                 classes: Sequence[str],
                 iou_thresholds: Sequence[float] = (0.1, 0.5),
                 fpi_thresholds: Sequence[float] = (1/8, 1/4, 1/2, 1, 2, 4, 8),
                 per_class: bool = False, verbose: bool = True,
                 save_dir: Optional[Union[str, Path]] = None,
                 ):
        """
        Class to compute FROC
        
        Multiclass FROC: This implementation performs the FROC over all
        objects regardless of their class which assigns each object the
        same "weight".
        
        Note this implementation is experimental and might change in the
        future. Please prefer the AP metric for now.

        Args:
            classes: name of each class
                (index needs to correspond to predicted class indices!)
            iou_thresholds: IoU thresholds for which FROC
                is evaluated
            fpi_thresholds: false positive per image
                thresholds (curve is interpolated at these values, score is
                the mean of the computed sens values at these positions)
            per_class: additional FROC curves are computed per class
            verbose: log time needed for evaluation
        """
        self.classes = classes
        self.iou_thresholds = iou_thresholds
        self.fpi_thresholds = fpi_thresholds
        self.per_class = per_class
        self.verbose = verbose

        if save_dir is None:
            self.save_dir = save_dir
        else:
            self.save_dir = Path(save_dir)

    def get_iou_thresholds(self) -> Sequence[float]:
        """
        Return IoU thresholds needed for this metric in an numpy array

        Returns:
            Sequence[float]: IoU thresholds [M], M is the number of thresholds
        """
        return self.iou_thresholds

    def compute(self, results_list: List[Dict[int, Dict[str, np.ndarray]]]) -> Tuple[
            Dict[str, float], Dict[str, np.ndarray]]:
        """
        Compute FROC

        Args:
            results_list: list with result s per image (in list)
                per category (dict). Inner Dict contains multiple results
                    obtained by :func:`box_matching_batch`.
                `dtMatches`: matched detections [T, D], where T = number of
                    thresholds, D = number of detections
                `gtMatches`: matched ground truth boxes [T, G], where
                    T = number of thresholds, G = number of  ground truth
                `dtScores`: prediction scores [D] detection scores
                `gtIgnore`: ground truth boxes which should be ignored
                    [G] indicate whether ground truth should be ignored
                `dtIgnore`: detections which should be ignored [T, D],
                    indicate which detections should be ignored

        Returns:
            Dict[str, float]: FROC score per IoU (key: FROC_score@IoU:{key:2f})
            Dict[str, np.ndarray]: FROC curve computed at specified fps
                thresholds per IoU; [R] R is the number of fps thresholds
                (key: FROC_curve@IoU:{key:2f})
        """
        if self.verbose:
            logger.info('Start FROC metric computation...')
            tic = time.time()

        scores = {}
        curves = {}
        _score, _curve = self.compute_froc_mul_iou(results_list)
        scores.update(_score)
        curves.update(_curve)

        if self.verbose:
            toc = time.time()
            logger.info(f'FROC finished (t={(toc - tic):0.2f}s).')

        if self.per_class:
            _score, _curve = self.compute_froc_mul_iou_per_class(results_list)
            scores.update(_score)
            curves.update(_curve)

            if self.verbose:
                toc = time.time()
                logger.info(f'FROC per class finished (t={(toc - tic):0.2f}s).')

        if self.save_dir is not None:
            self.plot_froc_curves(curves)
        return scores, curves

    def compute_froc_mul_iou(self, results_list: List[Dict[int, Dict[str, np.ndarray]]]) -> Tuple[
            Dict[str, float], Dict[str, np.ndarray]]:
        """
        Compute FROC curve for multiple IoU values

        Args:
            results_list: list with result s per image (in list)
                per category (dict). Inner Dict contains multiple results
                    obtained by :func:`box_matching_batch`.
                `dtMatches`: matched detections [T, G], where T = number of
                    thresholds, G = number of ground truth
                `gtMatches`: matched ground truth boxes [T, D], where
                    T = number of thresholds, D = number of detections
                `dtScores`: prediction scores [D] detection scores
                `gtIgnore`: ground truth boxes which should be ignored
                    [G] indicate whether ground truth should be ignored
                `dtIgnore`: detections which should be ignored [T, D],
                    indicate which detections should be ignored

        Returns:
            Dict[str, float]: FROC score per IoU
            Dict[str,np.ndarray]: FROC curve computed at specified fps 
                thresholds per IoU; [R] R is the number of fps thresholds
        """
        num_images = len(results_list)
        results = [_r for r in results_list for _r in r.values()]

        if len(results) == 0:
            logger.warning(f"WARNING, no results found for froc computation")
            return ({"froc_score": 0},
                    {"froc_curve": np.zeros(len(self.fpi_thresholds))})

        # r['dtMatches'] [T, R], where R = sum(all detections)
        dt_matches = np.concatenate([r['dtMatches'] for r in results], axis=1)
        dt_ignores = np.concatenate([r['dtIgnore'] for r in results], axis=1)
        dt_scores = np.concatenate([r['dtScores'] for r in results])
        gt_ignore = np.concatenate([r['gtIgnore'] for r in results])
        
        self.check_number_of_iou(dt_matches, dt_ignores)
        
        num_gt = np.count_nonzero(gt_ignore == 0)  # number of ground truth boxes (non ignored)
        if num_gt == 0:
            logger.error("No ground truth found! Returning 0 in FROC.")
            return ({"froc_score": 0},
                    {"froc_curve": np.zeros(len(self.fpi_thresholds))})

        # keep shape in case of 1 threshold
        old_shape = dt_matches.shape
        dt_matches = dt_matches[np.logical_not(dt_ignores)].reshape(old_shape)
        
        curves = {}
        for iou_idx, iou_val in enumerate(self.iou_thresholds):
            # filter scores with ignores detections
            _scores = dt_scores[np.logical_not(dt_ignores[iou_idx])]
            assert len(_scores) == len(dt_matches[iou_idx])
            
            _fps, _sens, _th = (self.compute_froc_curve_one_iou(
                dt_matches[iou_idx], _scores, num_images, num_gt))
            
            # interpolate at defined fpr thresholds
            curves[iou_val] = np.interp(self.fpi_thresholds, _fps, _sens)


        # linearly interpolate curves for needed fps values
        scores = {f"FROC_score_IoU_{key:.2f}": np.mean(c) for key, c in curves.items()}
        curves = {f"FROC_curve_IoU_{key:.2f}": c for key, c in curves.items()}
        curves["FROC_fpi_thresholds"] = self.fpi_thresholds
        return scores, curves

    @staticmethod
    def compute_froc_curve_one_iou(dt_matches: np.ndarray, dt_scores: np.ndarray,
                                   num_images: int, num_gt: int):
        """
        Compute FROC curve for a single IoU value

        Args:
            dt_matches (np.ndarray): binary array indicating which bounding
                boxes have a large enough overlap with gt;
                [R] where R is the number of predictions
            dt_scores (np.ndarray): prediction score for each bounding box;
                [R] where R is the number of predictions
            num_images (int): number of images
            num_gt (int): number of ground truth bounding boxes

        Returns:
            np.ndarray: false positives per image
            np.ndarray: sensitivity
            np.ndarray: thresholds
        """
        num_detections = len(dt_matches)
        num_matched = np.sum(dt_matches)
        num_unmatched = num_detections - num_matched

        if dt_matches.size == 0:
            logger.warning("WARNING, no matches found.")
            return np.zeros((2,)), np.zeros((2,)), np.zeros((2,))
        else:
            fpr, tpr, thresholds = roc_curve(dt_matches, dt_scores)

        if num_unmatched == 0:
            logger.warning("WARNING, no false positives found")
            fps = np.zeros(len(fpr))
        else:
            fps = (fpr * num_unmatched) / num_images
        sens = (tpr * num_matched) / num_gt
        return fps, sens, thresholds

    def compute_froc_mul_iou_per_class(
        self, results_list: List[Dict[int, Dict[str, np.ndarray]]]) -> (
            Dict[str, float], Dict[str, np.ndarray]):
        """
        Compute FROC curve for multiple classes

        Args:
            results_list: list with result s per image (in list)
                per category (dict). Inner Dict contains multiple results
                    obtained by :func:`box_matching_batch`.
                `dtMatches`: matched detections [T, G], where T = number of
                    thresholds, G = number of ground truth
                `gtMatches`: matched ground truth boxes [T, D], where
                    T = number of thresholds, D = number of detections
                `dtScores`: prediction scores [D] detection scores
                `gtIgnore`: ground truth boxes which should be ignored
                    [G] indicate whether ground truth should be ignored
                `dtIgnore`: detections which should be ignored [T, D],
                    indicate which detections should be ignored

        Returns:
            Dict[str, float]: FROC score computed  per class per class
            Dict[str, np.ndarray]: FROC curve computed per class per IoU;
                [R] R is the number of fps thresholds
        """
        froc_scores_cls = {}
        froc_curves_cls = {}
        for cls_idx, cls_str in enumerate(self.classes):
            # filter current class from list of results and put them into a dict with a single entry
            results_by_cls = [{0: r[cls_idx]} if cls_idx in r else {} for r in results_list]
            if results_by_cls:
                cls_scores, cls_curves = self.compute_froc_mul_iou(results_by_cls)

                froc_scores_cls.update({f"{cls_str}_{key}": item for key, item in cls_scores.items()})
                froc_curves_cls.update({f"{cls_str}_{key}": item for key, item in cls_curves.items()})
        return froc_scores_cls, froc_curves_cls

    def plot_froc_curves(self, curves: Dict[str, Sequence[float]]) -> None:
        """
        Plot frocs

        Args:
            curves: dict with froc curves (as obtained by :method:`compute`)
                FROC_score_IoU_{key:.2f} for class "normal" FROC
                {cls_name}_FROC_score_IoU_{key:.2f}: for class specific froc
        """
        # plot normal froc curves
        selection = select_froc_curves(curves)
        fig, ax = get_froc_ax(self.fpi_thresholds)
        for _, froc, iou in zip(*selection):
            ax.plot(self.fpi_thresholds, froc, 'o-', label=f"IoU:{iou:.2f}")
        ax.set_title("FROC")
        ax.legend(loc='lower right')
        fig.savefig(self.save_dir / "FROC.png")
        plt.close(fig)

        # plot cls frocs
        selection = select_froc_curves_cls(curves)
        reordered = defaultdict(list)
        for class_name, (names, frocs, ious) in selection.items():
            for froc, iou in zip(frocs, ious):
                reordered[iou].append((class_name, froc))
        for iou, frocs in reordered.items():
            fig, ax = get_froc_ax(self.fpi_thresholds)
            for class_name, froc in frocs:
                ax.plot(self.fpi_thresholds, froc, 'o-', label=f"{class_name}")
            title = f"FROC_cls_IoU_{iou:.2f}"
            ax.set_title(title)
            ax.legend(loc='lower right')
            fig.savefig(self.save_dir / f"{title.replace('.', '_')}.png")
            plt.close(fig)


def get_froc_ax(fpi_values: Optional[Sequence[float]] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create preconfigured figure and axes object for froc curves

    Args:
        fpi_values: x values to use for froc

    Returns:
        plt.Figure: figure object
        plt.Axes: configured axes object
    """
    fig, ax = plt.subplots()
    ax.set_xscale("log", base=2)
    
    if fpi_values is not None:
        ax.set_xlim(min(fpi_values), max(fpi_values))
        ax.set_xticks(fpi_values)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Avg number of false positives per scan')
    ax.set_ylabel('Sensitivity')
    ax.grid(True)

    formatter = FuncFormatter(lambda y, _: '{:.3f}'.format(y))
    ax.xaxis.set_major_formatter(formatter)
    return fig, ax


def select_froc_curves(curves: Dict[str, np.ndarray], prefix: Optional[str] = None) -> \
        Tuple[List[str], List[np.ndarray], List[float]]:
    """
    Select froc curves

    Args:
        curves: dict to select frocs from. Class specific frocs need to
            follow FROC_score_IoU_{key:.2f} pattern

    Returns:
        Dict[str, Tuple[List[str], List[np.ndarray], List[float]]]:
            dict defines the classes, tuple is output from
            :method:`select_froc_curves_cls`
    """
    if prefix is None:
        prefix = ""
    froc_keys = [str(c) for c in curves.keys()
                 if str(c).startswith(f"{prefix}FROC_") and
                 not str(c).endswith("_thresholds")]
    frocs = [curves[c] for c in froc_keys]
    ious = [float(c.rsplit('_', 1)[1]) for c in froc_keys]
    return froc_keys, frocs, ious


def select_froc_curves_cls(curves: Dict[str, np.ndarray]) -> \
        Dict[str, Tuple[List[str], List[np.ndarray], List[float]]]:
    """
    Select class specific froc curves

    Args:
        curves: dict to select frocs from. Class specific frocs need to follow
            {cls_name}_FROC_score_IoU_{key:.2f} pattern

    Returns:
        Dict[str, Tuple[List[str], List[np.ndarray], List[float]]]:
            dict defines the classes, tuple is output from
            :method:`select_froc_curves_cls`
    """
    all_classes = [str(c).split('_', 1)[0] for c in curves.keys()
                   if not str(c).startswith("FROC_") and
                   not str(c).endswith("_thresholds")]
    all_classes = list(set(all_classes))
    output = {}
    for cls_name in all_classes:
        output[cls_name] = select_froc_curves(curves, prefix=f"{cls_name}_")
    return output
