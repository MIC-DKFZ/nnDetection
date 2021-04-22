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

from pathlib import Path
from loguru import logger
from typing import Sequence, List, Dict, Any, Tuple


import matplotlib.pyplot as plt

from nndet.evaluator import DetectionMetric


class PredictionHistogram(DetectionMetric):
    def __init__(self,
                 classes: Sequence[str], save_dir: Path,
                 iou_thresholds: Sequence[float] = (0.1, 0.5),
                 bins: int = 50):
        """
        Class to compute prediction histograms. (Note: this class does not
        provide any scalar metrics)

        Args:
            classes: name of each class (index needs to correspond to predicted class indices!)
            save_dir: directory where histograms are saved to
            iou_thresholds: IoU thresholds for which FROC is evaluated
            bins: number of bins of histogram
        """
        self.classes = classes
        self.save_dir = save_dir

        self.iou_thresholds = iou_thresholds
        self.bins = bins

    def get_iou_thresholds(self) -> Sequence[float]:
        """
        Return IoU thresholds needed for this metric in an numpy array

        Returns:
            Sequence[float]: IoU thresholds [M], M is the number of thresholds
        """
        return self.iou_thresholds

    def compute(self, results_list: List[Dict[int, Dict[str, np.ndarray]]]) -> Tuple[
            Dict[str, float], Dict[str, Dict[str, Any]]]:
        """
        Plot class independent and per class histograms. For more info see
        `method``plot_hist`

        Args:
            Dict: results over dataset
        """
        self.plot_hist(results_list=results_list)
        for cls_idx, cls_str in enumerate(self.classes):
            # filter current class from list of results and put them into a dict with a single entry
            results_by_cls = [{0: r[cls_idx]} for r in results_list if cls_idx in r if cls_idx in r]
            self.plot_hist(results_by_cls, title_prefix=f"cl_{cls_str}_")
        return {}, {}

    def plot_hist(self, results_list: List[Dict[int, Dict[str, np.ndarray]]],
                  title_prefix: str = "") -> Tuple[
                    Dict[str, float], Dict[str, Dict[str, Any]]]:
        """
        Compute prediction histograms for multiple IoU values

        Args:
            results_list (List[Dict[int, Dict[str, np.ndarray]]]): list with result s per image (in list)
                per category (dict). Inner Dict contains multiple results obtained by :func:`box_matching_batch`.
                `dtMatches`: matched detections [T, G], where T = number of thresholds, G = number of ground truth
                `gtMatches`: matched ground truth boxes [T, D], where T = number of thresholds,
                    D = number of detections
                `dtScores`: prediction scores [D] detection scores
                `gtIgnore`: ground truth boxes which should be ignored [G] indicate whether ground truth
                    should be ignored
                `dtIgnore`: detections which should be ignored [T, D], indicate which detections should be ignored
            title_prefix: prefix for title of histogram plot

        Returns:
            Dict: empty
            Dict[Dict[str, Any]]: histogram informations
                `{IoU Value}`:
                    `tp_hist` (np.ndarray): histogram if true positives; false negatives @ score=0 [:attr:`self.bins`]
                    `fp_hist` (np.ndarray): false positive histogram [:attr:`self.bins`]
                    `true_positives` (int): number of true positives according to matching
                    `false_positives` (int): number of false_positives according to matching
                    `false_negatives` (int): number of false_negatives according to matching
        """
        num_images = len(results_list)
        results = [_r for r in results_list for _r in r.values()]

        if len(results) == 0:
            logger.warning(f"WARNING, no results found for froc computation")
            return {}, {}

        # r['dtMatches'] [T, R], where R = sum(all detections)
        dt_matches = np.concatenate([r['dtMatches'] for r in results], axis=1)
        dt_ignores = np.concatenate([r['dtIgnore'] for r in results], axis=1)
        dt_scores = np.concatenate([r['dtScores'] for r in results])
        gt_ignore = np.concatenate([r['gtIgnore'] for r in results])
        self.check_number_of_iou(dt_matches, dt_ignores)
        
        num_gt = np.count_nonzero(gt_ignore == 0)  # number of ground truth boxes (non ignored)
        if num_gt == 0:
            logger.error("No ground truth found! Returning nothing.")
            return {}, {}

        for iou_idx, iou_val in enumerate(self.iou_thresholds):
            # filter scores with ignores detections
            _scores = dt_scores[np.logical_not(dt_ignores[iou_idx])]
            assert len(_scores) == len(dt_matches[iou_idx])
            _ = self.compute_histogram_one_iou(\
                dt_matches[iou_idx], _scores, num_images, num_gt, iou_val, title_prefix)
        return {}, {}

    def compute_histogram_one_iou(self, dt_matches: np.ndarray, dt_scores: np.ndarray,
                                  num_images: int, num_gt: int, iou: float,
                                  title_prefix: str):
        """
        Plot prediction histogram
        
        Args:
            dt_matches (np.ndarray): binary array indicating which bounding
                boxes have a large enough overlap with gt;
                [R] where R is the number of predictions
            dt_scores (np.ndarray): prediction score for each bounding box;
                [R] where R is the number of predictions
            num_images (int): number of images
            num_gt (int): number of ground truth bounding boxes
            iou: IoU values which is currently evaluated
            title_prefix: prefix for title of histogram plot
        """
        num_matched = np.sum(dt_matches)
        false_negatives = num_gt - num_matched # false negatives
        true_positives = np.sum(dt_matches)
        false_positives = np.sum(dt_matches == 0)

        _dt_matches = np.concatenate([dt_matches, [1] * int(false_negatives)])
        _dt_scores = np.concatenate([dt_scores, [0] * int(false_negatives)])

        plt.figure()
        plt.yscale('log')
        if 0 in dt_matches:
            plt.hist(_dt_scores[_dt_matches == 0], bins=self.bins, range=(0., 1.), 
                    alpha=0.3, color='g', label='false pos.')
        if 1 in dt_matches:
            plt.hist(_dt_scores[_dt_matches == 1], bins=self.bins, range=(0., 1.),
                    alpha=0.3, color='b', label='true pos. (false neg. @ score=0)')
        plt.legend()
        title = title_prefix + (f"tp:{true_positives} fp:{false_positives} "
                                f"fn:{false_negatives} pos:{true_positives+false_negatives}")
        plt.title(title)
        plt.xlabel('confidence score')
        plt.ylabel('log n')

        if self.save_dir is not None:
            save_path = self.save_dir / (f"{title_prefix}pred_hist_IoU@{iou}".replace(".", "_") + ".png")
            logger.info(f"Saving {save_path}")
            plt.savefig(save_path)
        plt.close()
        return None
