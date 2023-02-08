# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import roc_curve

from nndet.evaluator import DetectionMetric


class FROCMetric(DetectionMetric):
    def __init__(
        self,
        classes: Sequence[str],
        iou_thresholds: Sequence[float] = (0.1, 0.5),
        fpi_thresholds: Sequence[float] = (1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8),
        verbose: bool = True,
    ):
        """
        Class to compute FROC

        Multiclass FROC: This implementation performs the FROC over all
        objects regardless of their class which assigns each object the
        same "weight".

        Update: Added support for equal class weighted FROC score.
        Curves are still only supported with pool version or for each class
        individually!

        Args:
            classes: name of each class
                (index needs to correspond to predicted class indices!)
            iou_thresholds: IoU thresholds for which FROC
                is evaluated
            fpi_thresholds: false positive per image
                thresholds (curve is interpolated at these values, score is
                the mean of the computed sens values at these positions)
            verbose: log time needed for evaluation

        Notes:
            c_FROC_num_images should stay constant across all classes since the
            number of images doesn't change across the data set.
        """
        self.classes = classes
        self.iou_thresholds = iou_thresholds
        self.fpi_thresholds = fpi_thresholds
        self.verbose = verbose

    def get_save_name(self, tag: Optional[str] = None) -> str:
        """
        Return name of file to save

        Returns:
            str: Name of the Metric and the chosen setting
        """
        return f"FROC_{tag}" if tag is not None else "FROC"

    def get_iou_thresholds(self) -> Sequence[float]:
        """
        Return IoU thresholds needed for this metric in an numpy array

        Returns:
            Sequence[float]: IoU thresholds [M], M is the number of thresholds
        """
        return self.iou_thresholds

    def compute(
        self,
        results_list: List[Dict[int, Dict[str, np.ndarray]]],
        tag: Optional[str] = None,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Compute FROC

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
            Dict[str, float]: FROC score per IoU (key: FROC_score@IoU:{key:2f})
            Dict[str, np.ndarray]: FROC curve computed at specified fps
                thresholds per IoU; [R] R is the number of fps thresholds
                (key: FROC_curve@IoU:{key:2f})
        """
        if self.verbose:
            logger.info("Start FROC metric computation...")
            tic = time.time()

        scores = {}
        curves = {
            "iou_thresholds": list(self.iou_thresholds),
            "fpi_thresholds": self.fpi_thresholds,
            "classes": self.classes,
            "save_name": self.get_save_name(tag=tag),
        }
        _score, _curve = self.compute_froc_mul_iou(results_list, tag=tag)
        scores.update(_score)
        curves.update(_curve)

        if self.verbose:
            toc = time.time()
            logger.info(f"FROC finished (t={(toc - tic):0.2f}s).")

        _score, _curve = self.compute_froc_mul_iou_per_class(results_list, tag=tag)
        scores.update(_score)
        curves.update(_curve)

        if self.verbose:
            toc = time.time()
            logger.info(f"FROC per class finished (t={(toc - tic):0.2f}s).")
        return scores, curves

    def compute_froc_mul_iou(
        self,
        results_list: List[Dict[int, Dict[str, np.ndarray]]],
        tag: Optional[str],
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Compute FROC curve for multiple IoU values

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
            Dict[str, float]: FROC score per IoU
            Dict[str,np.ndarray]: FROC curve computed at specified fps
                thresholds per IoU; [R] R is the number of fps thresholds
        """
        num_images = len(results_list)
        results = [_r for r in results_list for _r in r.values()]

        save_name = self.get_save_name(tag=tag)
        if len(results) == 0:
            logger.warning(f"No results found for {self.get_save_name(tag=tag)}")
            return self.zero_result(
                save_name=save_name,
                num_images=num_images,
                num_gt=0,
            )

        # r['dtMatches'] [T, R], where R = sum(all detections)
        dt_matches = np.concatenate([r["dtMatches"] for r in results], axis=1)
        dt_ignores = np.concatenate([r["dtIgnore"] for r in results], axis=1)
        dt_scores = np.concatenate([r["dtScores"] for r in results])
        gt_ignore = np.concatenate([r["gtIgnore"] for r in results])

        self.check_number_of_iou(dt_matches, dt_ignores)

        num_gt = np.count_nonzero(gt_ignore == 0)  # number of ground truth boxes (non ignored)
        if num_gt == 0:
            logger.debug(f"No gt found for {self.get_save_name(tag=tag)}")
            return self.zero_result(
                save_name=save_name,
                num_images=num_images,
                num_gt=num_gt,
            )

        scores = {}
        meta = {
            "num_images": num_images,
            "num_gt": num_gt,
        }
        for iou_idx, iou_val in enumerate(self.iou_thresholds):
            # filter scores and matches with detection ignores
            _scores = dt_scores[np.logical_not(dt_ignores[iou_idx])]
            _dt_matches = dt_matches[iou_idx][np.logical_not(dt_ignores[iou_idx])]
            assert len(_scores) == len(_dt_matches)

            _fps, _sens, _th = self.compute_froc_curve_one_iou(_dt_matches, _scores, num_images, num_gt)

            # interpolate at defined fpr thresholds
            sens_interp = np.interp(self.fpi_thresholds, _fps, _sens)
            scores[f"{save_name}_IoU_{iou_val:.2f}"] = np.mean(sens_interp)
            meta[f"{save_name}_IoU_{iou_val:.2f}"] = sens_interp
        return scores, meta

    def compute_froc_mul_iou_per_class(
        self,
        results_list: List[Dict[int, Dict[str, np.ndarray]]],
        tag: Optional[str],
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Compute FROC curve for multiple classes

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
            Dict[str, float]: FROC score computed  per class per class
            Dict[str, np.ndarray]: FROC curve computed per class per IoU;
                [R] R is the number of fps thresholds
        """
        froc_scores_cls = {}
        froc_curves_cls = {}
        froc_scores_cache = defaultdict(list)  # per metric cache
        for cls_idx, cls_str in enumerate(self.classes):
            # filter current class from list of results and put them into a dict with a single entry
            num_images_og = len(results_list)
            results_by_cls = [{0: r[cls_idx]} if cls_idx in r else {} for r in results_list]
            assert len(results_by_cls) == num_images_og, "Inconsistent num images!"
            if results_by_cls:
                cls_scores, cls_curves = self.compute_froc_mul_iou(results_by_cls, tag=tag)

                for key, item in cls_scores.items():
                    froc_scores_cache[key].append(item)

                froc_scores_cls.update({f"{cls_str}_{key}": item for key, item in cls_scores.items()})
                froc_curves_cls.update({f"{cls_str}_{key}": item for key, item in cls_curves.items()})

        for metric_str, metric_cache in froc_scores_cache.items():
            froc_scores_cls[f"mc_{metric_str}"] = float(sum(metric_cache) / len(self.classes))

        return froc_scores_cls, froc_curves_cls

    def zero_result(
        self,
        save_name: str,
        num_images: int,
        num_gt: int,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Helper function to create zero result

        Args:
            save_name: name of evaluation
            num_images: number of images
            num_gt: number of ground truth objects

        Returns:
            Dict[str, float]: FROC score per IoU
            Dict[str,np.ndarray]: FROC curve computed at specified fps
                thresholds per IoU; [R] R is the number of fps thresholds
        """
        scores = {}
        curves = {
            "num_images": num_images,
            "num_gt": num_gt,
        }
        for _, iou_val in enumerate(self.iou_thresholds):
            scores[f"{save_name}_IoU_{iou_val:.2f}"] = np.nan
            curves[f"{save_name}_IoU_{iou_val:.2f}"] = np.zeros(len(self.fpi_thresholds))
        return scores, curves

    @staticmethod
    def compute_froc_curve_one_iou(
        dt_matches: np.ndarray,
        dt_scores: np.ndarray,
        num_images: int,
        num_gt: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute FROC curve for a single IoU value

        Args:
            dt_matches: binary array indicating which bounding
                boxes have a large enough overlap with gt;
                [R] where R is the number of predictions
            dt_scores: prediction score for each bounding box;
                [R] where R is the number of predictions
            num_images: number of images
            num_gt: number of ground truth bounding boxes

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

    @classmethod
    def plot(
        cls,
        result_scores: Dict[str, float],
        result_meta: Dict[str, Any],
        save_dir: os.PathLike,
    ) -> None:
        """
        Plot FROC curves
        (these are alrady interpolated!)

        Args:
            result_scores: single as obtained from `compute` function
            result_meta: meta information as obtained from `compute` function
            save_dir: path to directory where files should be saved

        Returns:
            Dict: figures of create plots
        """
        fpi = result_meta["fpi_thresholds"]
        save_name = result_meta["save_name"]
        for iou in result_meta["iou_thresholds"]:
            # parse info
            froc_score_pool = result_scores[f"{save_name}_IoU_{iou:.2f}"]
            froc_score_mc = result_scores[f"mc_{save_name}_IoU_{iou:.2f}"]
            num_images = result_meta["num_images"]

            # create plot
            fig, ax = get_froc_ax()
            for cls_str in result_meta["classes"]:
                key = f"{cls_str}_{save_name}_IoU_{iou:.2f}"
                sens = result_meta[key]
                num_objects = result_meta[f"{cls_str}_num_gt"]
                ax.plot(fpi, sens, "o-", label=f"{cls_str} FROC {result_scores[key]:.2f} N={num_objects}")

            title = f"{save_name}_IoU_{iou:.2f}"
            ax.set_title(f"{title}: Pool {froc_score_pool:.2f} MC {froc_score_mc:.2f} \n" f"Num images: {num_images}")
            ax.legend(loc="lower right")

            # save file
            ap_save_dir = Path(save_dir) / "results_FROC"
            ap_save_dir.mkdir(exist_ok=True)
            # fig.savefig(ap_save_dir / f"{title.replace('.', '_')}.png")
            fig.savefig(ap_save_dir / f"{title.replace('.', '_')}.pdf")
            plt.close(fig)


def get_froc_ax(
    fpi_values: Optional[Sequence[float]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
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
    # ax.set_xscale("linear")

    if fpi_values is not None:
        ax.set_xlim(min(fpi_values), max(fpi_values))
        ax.set_xticks(fpi_values)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Avg number of false positives per scan")
    ax.set_ylabel("Sensitivity")
    ax.grid(True)

    formatter = FuncFormatter(lambda y, _: "{:.3f}".format(y))
    ax.xaxis.set_major_formatter(formatter)
    return fig, ax
