# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

import numpy as np

from nndet.evaluator.det import MaskEvaluator
from nndet.inference.sweeper.boxes import BoxSweeper
from nndet.io.transforms.instances import instances_to_binary_masks_np
from nndet.utils import to_numpy
from nndet.utils.info import maybe_verbose_iterable


class MaskSweeper(BoxSweeper):
    evaluator_cls = MaskEvaluator

    def _evaluate_value(
        self,
        state: Dict[str, Any],
        **overwrite,
    ):
        """
        Evalaute a single value

        Args:
            state: state for ensembler
            overwrite: state overwrites

        Returns:
            Dict: scalar metrics
        """
        evaluator = self.evaluator_cls.create(
            classes=self.classes,
            fast=True,
            verbose=False,
            save_dir=None,
        )

        for case_id in maybe_verbose_iterable(
            self.ensembler_cls.get_case_ids(self.pred_dir)
        ):
            ensembler = self.ensembler_cls.from_checkpoint(
                base_dir=self.pred_dir,
                case_id=case_id,
                device=self.device,
            )
            ensembler.update_parameters(**state)
            ensembler.update_parameters(**overwrite)

            pred = to_numpy(ensembler.get_case_result(restore=False))
            gt = np.load(
                str(self.gt_dir / f"{case_id}_instances_gt.npz"), allow_pickle=True
            )
            # FIXME
            gt_boxes = np.load(
                str(self.gt_dir / f"{case_id}_boxes_gt.npz"), allow_pickle=True
            )

            pred_masks = pred["pred_masks"]
            if gt["instances"].ndim < (pred_masks.ndim - 1):
                gt_instances = gt["instances"][None]
            else:
                gt_instances = gt["instances"]

            evaluator.run_online_evaluation(
                pred_boxes=[pred_masks],
                pred_classes=[pred["pred_labels"]],
                pred_scores=[pred["pred_scores"]],
                gt_boxes=[instances_to_binary_masks_np(gt_instances)],
                gt_classes=[gt_boxes["classes"]],  # FIXME
                gt_ignore=None,
            )

        metric_scores, _ = evaluator.finish_online_evaluation()
        return metric_scores
