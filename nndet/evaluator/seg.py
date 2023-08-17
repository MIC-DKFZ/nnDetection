# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
from loguru import logger

from nndet.evaluator import AbstractEvaluator

__all__ = ["SegmentationEvaluator"]


class SegmentationEvaluator(AbstractEvaluator):
    def __init__(
        self,
        per_class: bool = True,
        *args,
        **kwargs,
    ):
        """
        Compute dice score during training

        Args:
            per_class: report per class dice scores
        """
        self.per_class = per_class
        self.results_list = defaultdict(list)

    def reset(self):
        """
        Reset internal state for new epoch
        """
        self.results_list = defaultdict(list)

    def run_online_evaluation(
        self,
        seg_probs: np.ndarray,
        target: np.ndarray,
    ) -> Dict:
        """
        Run evaluation of one batch and save internal results for later

        Args:
            seg_probs: output probabilities of network [N, C, dims], where N
                is the batch size, C is the number of classes, dims are
                spatial dimensions
            target: ground truth segmentation [N, dims], where N is the batch
                size and dims are spatial dimensions

        Returns:
            Dict: empty dict
        """
        num_classes = seg_probs.shape[1]
        output_seg = np.argmax(seg_probs, axis=1).reshape((seg_probs.shape[0], -1))
        target = target.reshape((target.shape[0], -1))

        tp_hard = np.zeros((target.shape[0], num_classes - 1))
        fp_hard = np.zeros((target.shape[0], num_classes - 1))
        fn_hard = np.zeros((target.shape[0], num_classes - 1))

        for c in range(1, num_classes):
            tp_hard[:, c - 1] = ((output_seg == c).astype(np.float32) * (target == c).astype(np.float32)).sum(axis=1)
            fp_hard[:, c - 1] = ((output_seg == c).astype(np.float32) * (target != c).astype(np.float32)).sum(axis=1)
            fn_hard[:, c - 1] = ((output_seg != c).astype(np.float32) * (target == c).astype(np.float32)).sum(axis=1)

        tp_hard = tp_hard.sum(axis=0)
        fp_hard = fp_hard.sum(axis=0)
        fn_hard = fn_hard.sum(axis=0)

        self.results_list["tp"].append(tp_hard)
        self.results_list["fp"].append(fp_hard)
        self.results_list["fn"].append(fn_hard)
        return {}

    def finish_online_evaluation(
        self,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Summarize results from batches and compute global dice and global
        dice per class

        Returns:
            Dict: results
                `{cls_idx}_seg_dice`: global dice per class
                `seg_dice`: global dice over all classes
        """
        results = {}
        if self.results_list:
            tp = np.sum(self.results_list["tp"], 0)
            fp = np.sum(self.results_list["fp"], 0)
            fn = np.sum(self.results_list["fn"], 0)

            global_dc_per_class = [
                i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)] if not np.isnan(i)
            ]
            if self.per_class:
                for cls_idx, dc in enumerate(global_dc_per_class):
                    results[f"{cls_idx}_seg_dice"] = dc
            results["seg_dice"] = np.mean(global_dc_per_class)
        else:
            logger.warning("No segmentation results found.")
        return results, None

    @classmethod
    def create(
        cls,
        per_class: bool = False,
        fg_mode: bool = False,
    ):
        return cls(
            per_class=per_class,
            fg_mode=fg_mode,
        )
