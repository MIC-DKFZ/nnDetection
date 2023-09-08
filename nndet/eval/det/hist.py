# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from nndet.eval import DetectionMetric


class PredictionHistogram(DetectionMetric):
    def __init__(
        self,
        classes: Sequence[str],
        save_dir: Path,
        iou_thresholds: Sequence[float] = (0.1, 0.5),
        bins: int = 50,
    ):
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
        self.value_range = (0, 1)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(classes: {self.classes}, iou_thresholds: {self.iou_thresholds}, "
            f"bins: {self.bins})"
        )

    @staticmethod
    def get_name(tag: Optional[str] = None) -> str:
        """
        Return name of file to save

        Returns:
            str: Name of the Metric and the chosen setting
            str: Tag Prefix for meta information
        """
        return f"pred_hist_{tag}" if tag is not None else "pred_hist"

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
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
        """
        Plot class independent and per class histograms. For more info see
        `method``plot_hist`

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
        """
        metric_name = self.get_name(tag=tag)
        results = {
            f"{metric_name}_iou_thresholds": self.iou_thresholds,
            f"{metric_name}_bins": self.bins,
            f"{metric_name}_value_range": self.value_range,
            f"{metric_name}_bin_edges": np.histogram([], bins=self.bins, range=self.value_range)[1],
            f"{metric_name}_classes": self.classes,
        }
        _, curves = self.compute_hist(results_list=results_list, tag=tag)
        results.update(curves)

        for cls_idx, cls_str in enumerate(self.classes):
            # filter current class from list of results and put them into a dict with a single entry
            results_by_cls = [{0: r[cls_idx]} if cls_idx in r else {} for r in results_list]
            _, cls_curves = self.compute_hist(results_list=results_by_cls, tag=tag)
            for key, item in cls_curves.items():
                results[f"{cls_str}_{key}"] = item
        return {}, results

    def compute_hist(
        self,
        results_list: List[Dict[int, Dict[str, np.ndarray]]],
        tag: Optional[str],
    ) -> Tuple[Dict, Dict]:
        """
        Compute prediction histograms for multiple IoU values

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
            Dict: empty
            Dict[Dict[str, Any]]: histogram informations

                ``'{metric_name}_IoU_{iou_val:.2f}_counts_tp'`` np.ndarray
                    histogram counts for matched predictions

                ``'{metric_name}_IoU_{iou_val:.2f}_counts_fp'`` np.ndarray
                    histogram counts for unmatched predictions

                ``'{metric_name}_IoU_{iou_val:.2f}_tp'`` int
                    number of matched predictions

                ``'{metric_name}_IoU_{iou_val:.2f}_fp'`` int
                    number of unmatched predictions

                ``'{metric_name}_IoU_{iou_val:.2f}_fn'`` int
                    number of umatched ground truth
        """
        metric_name = self.get_name(tag=tag)
        results = [_r for r in results_list for _r in r.values()]

        if len(results) == 0:
            logger.warning(f"No results found for {metric_name}")
            return {}, self.zero_result(metric_name=metric_name)

        # r['dtMatches'] [T, R], where R = sum(all detections)
        dt_matches = np.concatenate([r["dtMatches"] for r in results], axis=1)
        dt_ignores = np.concatenate([r["dtIgnore"] for r in results], axis=1)
        dt_scores = np.concatenate([r["dtScores"] for r in results])
        gt_ignore = np.concatenate([r["gtIgnore"] for r in results])
        self.check_number_of_iou(dt_matches, dt_ignores)

        num_gt = np.count_nonzero(gt_ignore == 0)  # number of ground truth boxes (non ignored)

        results = {}
        for iou_idx, iou_val in enumerate(self.iou_thresholds):
            # filter scores and matches with detection ignores
            _scores = dt_scores[np.logical_not(dt_ignores[iou_idx])]
            _dt_matches = dt_matches[iou_idx][np.logical_not(dt_ignores[iou_idx])]

            num_matched = np.sum(_dt_matches)
            false_negatives = num_gt - num_matched  # false negatives
            true_positives = np.sum(_dt_matches == 1)
            false_positives = np.sum(_dt_matches == 0)

            _dt_matches_with_fn = np.concatenate([_dt_matches, [1] * int(false_negatives)])
            _dt_scores_with_fn = np.concatenate([_scores, [0] * int(false_negatives)])

            counts_tp, _ = np.histogram(
                _dt_scores_with_fn[_dt_matches_with_fn == 1], bins=self.bins, range=self.value_range
            )
            counts_fp, _ = np.histogram(
                _dt_scores_with_fn[_dt_matches_with_fn == 0], bins=self.bins, range=self.value_range
            )

            results[f"{metric_name}_IoU_{iou_val:.2f}_counts_tp"] = counts_tp
            results[f"{metric_name}_IoU_{iou_val:.2f}_counts_fp"] = counts_fp
            results[f"{metric_name}_IoU_{iou_val:.2f}_tp"] = true_positives
            results[f"{metric_name}_IoU_{iou_val:.2f}_fp"] = false_positives
            results[f"{metric_name}_IoU_{iou_val:.2f}_fn"] = false_negatives
        return {}, results

    def zero_result(self, metric_name: str) -> Dict[str, np.ndarray]:
        """
        Create results with all zeros

        Args:
            metric_name: tagged metric name

        Returns:
            Dict[str, np.ndarray]: computed histogram counts
        """
        results = {}
        for _, iou_val in enumerate(self.iou_thresholds):
            results[f"{metric_name}_IoU_{iou_val:.2f}_counts_fp"] = np.zeros(self.bins)
            results[f"{metric_name}_IoU_{iou_val:.2f}_counts_tp"] = np.zeros(self.bins)
            results[f"{metric_name}_IoU_{iou_val:.2f}_tp"] = 0
            results[f"{metric_name}_IoU_{iou_val:.2f}_fp"] = 0
            results[f"{metric_name}_IoU_{iou_val:.2f}_fn"] = 0
        return results

    @classmethod
    def plot(
        cls,
        result_scores: Dict[str, float],
        result_meta: Dict[str, Any],
        save_dir: os.PathLike,
        tag: Optional[str] = None,
    ) -> None:
        """
        Plot Histograms

        Args:
            result_scores: single as obtained from `compute` function
            result_meta: meta information as obtained from `compute` function
            save_dir: path to directory where files should be saved
            tag: tag used during computation

        Returns:
            Dict: figures of create plots
        """
        metric_name = cls.get_name(tag=tag)
        edges = result_meta[f"{metric_name}_bin_edges"]

        hist_save_dir = Path(save_dir) / "results_histogram"
        hist_save_dir.mkdir(exist_ok=True)

        for iou in result_meta[f"{metric_name}_iou_thresholds"]:
            # iou histogram
            save_title = f"{metric_name}_IoU_{iou}"
            fig, ax = cls.histogram(
                edges=edges,
                counts_tp=result_meta[f"{metric_name}_IoU_{iou:.2f}_counts_tp"],
                counts_fp=result_meta[f"{metric_name}_IoU_{iou:.2f}_counts_fp"],
                tp=result_meta[f"{metric_name}_IoU_{iou:.2f}_tp"],
                fp=result_meta[f"{metric_name}_IoU_{iou:.2f}_fp"],
                fn=result_meta[f"{metric_name}_IoU_{iou:.2f}_fn"],
                title_prefix=save_title,
            )
            fig.savefig(hist_save_dir / f"{save_title.replace('.', '_')}.pdf")
            plt.close(fig)

            # per class histograms
            for cls_str in result_meta[f"{metric_name}_classes"]:
                save_title = f"{cls_str}_{metric_name}_IoU_{iou}"
                fig, ax = cls.histogram(
                    edges=edges,
                    counts_tp=result_meta[f"{cls_str}_{metric_name}_IoU_{iou:.2f}_counts_tp"],
                    counts_fp=result_meta[f"{cls_str}_{metric_name}_IoU_{iou:.2f}_counts_fp"],
                    tp=result_meta[f"{cls_str}_{metric_name}_IoU_{iou:.2f}_tp"],
                    fp=result_meta[f"{cls_str}_{metric_name}_IoU_{iou:.2f}_fp"],
                    fn=result_meta[f"{cls_str}_{metric_name}_IoU_{iou:.2f}_fn"],
                    title_prefix=save_title,
                )
                fig.savefig(hist_save_dir / f"{save_title.replace('.', '_')}.pdf")
                plt.close(fig)

    @staticmethod
    def histogram(
        edges: np.ndarray,
        counts_tp: np.ndarray,
        counts_fp: np.ndarray,
        tp: int,
        fp: int,
        fn: int,
        title_prefix: str,
    ):
        """
        Helper function to plot histograms
        """
        fig, ax = plt.subplots()
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("confidence score")
        ax.set_ylabel("log n")
        ax.set_yscale("log")
        ax.grid(True)

        ax.stairs(
            counts_fp,
            edges,
            alpha=0.2,
            color="k",
            fill=True,
            label="false pos",
        )
        ax.stairs(
            counts_tp,
            edges,
            alpha=0.3,
            color="g",
            fill=True,
            label="true pos. (false neg. @ score=0)",
        )
        title = f"{title_prefix}\ntp:{tp} fp:{fp} fn:{fn} pos:{tp+fn}"
        ax.set_title(title)
        ax.legend(loc="upper center")
        return fig, ax
