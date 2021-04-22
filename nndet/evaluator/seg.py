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

import numpy as np
from typing import Dict, Sequence, Tuple
from collections import defaultdict

from nndet.evaluator import AbstractEvaluator


__all__ = ["SegmentationEvaluator"]


class SegmentationEvaluator(AbstractEvaluator):
    def __init__(self,
                 per_class: bool = True,
                 *args,
                 **kwargs,
                 ):
        """
        Compute dice score during training
        """
        self.per_class = per_class
        self.results_list = defaultdict(list)

    def reset(self):
        """
        Reset internal state for new epoch
        """
        self.results_list = defaultdict(list)

    def run_online_evaluation(self,
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

        self.results_list["fg_dice"] = list(
            (2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8))
        self.results_list["tp"].append(tp_hard)
        self.results_list["fp"].append(fp_hard)
        self.results_list["fn"].append(fn_hard)
        return {}

    def finish_online_evaluation(self) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
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
                i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)] if not np.isnan(i)]
            if self.per_class:
                for cls_idx, dc in enumerate(global_dc_per_class):
                    results[f"{cls_idx}_seg_dice"] = dc
            results["seg_dice"] = np.mean(global_dc_per_class)
        return results, None

    @classmethod
    def create(cls,
               per_class: bool = False,
               ):
        return cls(per_class=per_class)


class PerCaseSegmentationEvaluator(AbstractEvaluator):
    def __init__(self,
                 classes: Sequence[str],
                 *args,
                 **kwargs,
                 ):
        """
        Compute dice score per case and average results over dataset
        """
        self.classes = classes
        self.results = []

    def reset(self):
        """
        Reset internal state for new epoch
        """
        self.results = []

    def run_online_evaluation(self,
                              seg: np.ndarray,
                              target: np.ndarray,
                              ) -> Dict:
        """
        Run evaluation of one batch and save internal results for later

        Args:
            seg: output segmentation [N, dims]
            target: ground truth segmentation [N, dims], where N is the batch
                size and dims are spatial dimensions

        Returns:
            Dict: empty dict
        """
        assert len(seg) == len(target)

        num_classes = len(self.classes)
        output_seg = seg.reshape((seg.shape[0], -1)) # N, X
        target = target.reshape((target.shape[0], -1)) # N, X

        tp_hard = np.zeros((target.shape[0], num_classes - 1)) # N, FG
        fp_hard = np.zeros((target.shape[0], num_classes - 1)) # N ,FG
        fn_hard = np.zeros((target.shape[0], num_classes - 1)) # N, FG
        fg_present = np.zeros((target.shape[0], num_classes - 1)) # N, FG

        for c in range(1, num_classes):
            tp_hard[:, c - 1] = ((output_seg == c).astype(np.float32) * (target == c).astype(np.float32)).sum(axis=1)
            fp_hard[:, c - 1] = ((output_seg == c).astype(np.float32) * (target != c).astype(np.float32)).sum(axis=1)
            fn_hard[:, c - 1] = ((output_seg != c).astype(np.float32) * (target == c).astype(np.float32)).sum(axis=1)
            fg_present[:, c - 1] = (target == c).any(axis=1).astype(np.int32)

        dice = np.where(fg_present, 2. * tp_hard / (2 * tp_hard + fp_hard + fn_hard), np.nan) # N, FG
        self.results.append(dice)
        return {}

    def finish_online_evaluation(self) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Summarize results from batches and compute global dice and global
        dice per class

        Returns:
            Dict: results
                `{cls_idx}_seg_dice`: global dice per class
                `seg_dice`: global dice over all classes
        """
        dice_full = np.concatenate(self.results, axis=0)
        dice_per_class = dice_full.mean(axies=0) # C
        dice = dice_full.mean() # 1
        
        results = {}
        for cls_idx, value in enumerate(dice_per_class):
            results[f"dice_cls_{cls_idx}"] = float(value)
        results["dice"] = float(dice)
        return results, None
    
    @classmethod
    def create(cls,
               classes: Sequence[str],
               ):
        return cls(classes=classes)
