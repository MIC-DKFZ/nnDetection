# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import numpy as np
from loguru import logger

from nndet.evaluator import AbstractEvaluator
from nndet.evaluator.det import BoxEvaluator
from nndet.ptmodule.mixins.evaluation.base import EvalMixin
from nndet.utils.tensor import to_numpy


class BoxEvalMixin(EvalMixin):
    def evaluation_init(self, plan: dict) -> Dict[str, AbstractEvaluator]:
        """
        Initialize `BoxEvaluator`

        Notes:
        make sure to call the super classes here!
        """
        evaluators = super().evaluation_init(plan=plan)
        if "boxes" in evaluators:
            raise RuntimeError("Found BoxEvaluator in evaluators, can not register a second one!")

        _classes = [f"class{c}" for c in range(plan["architecture"]["classifier_classes"])]
        evaluators["boxes"] = BoxEvaluator.create(
            classes=_classes,
            fast=True,
            save_dir=None,
        )
        return evaluators

    def evaluation_step(
        self,
        predictions: dict,
        targets: dict,
    ) -> None:
        """
        Evaluate a validation batch with metrics

        Args:
        predictions: dict with predictions.
        Exact keys depend on the module class
        targets: dict with ground truth.
        Exact keys depend on the module class.

        Notes:
        make sure to call the super classes here!
        """
        super().evaluation_step(predictions=predictions, targets=targets)

        pred_boxes = to_numpy(predictions["pred_boxes"])
        pred_classes = to_numpy(predictions["pred_labels"])
        pred_scores = to_numpy(predictions["pred_scores"])

        gt_boxes = to_numpy(targets["target_boxes"])
        gt_classes = to_numpy(targets["target_classes"])
        gt_ignore = None

        assert len(pred_boxes) == len(gt_boxes)

        self.evaluators["boxes"].run_online_evaluation(
            pred_boxes=pred_boxes,
            pred_classes=pred_classes,
            pred_scores=pred_scores,
            gt_boxes=gt_boxes,
            gt_classes=gt_classes,
            gt_ignore=gt_ignore,
        )

    def evaluation_end(self) -> Dict[str, float]:
        """
        Compute validation metrics of epoch

        Notes:
        make sure to call the super classes here!
        """
        # collect other scores
        metric_scores = super().evaluation_end()

        # compute own scores
        box_scores, _ = self.evaluators["boxes"].finish_online_evaluation()
        self.evaluators["boxes"].reset()

        # add own scores
        metric_scores.update(box_scores)

        # [optional] log own scores
        logger.info(
            "Box::   "
            f"mAP@0.1:0.5:0.05: {box_scores.get('mAP_IoU_0.10_0.50_0.05_MaxDet_100', np.nan):0.3f}  "
            f"AP@0.1: {box_scores.get('AP_IoU_0.10_MaxDet_100', np.nan):0.3f}  "
            f"AP@0.5: {box_scores.get('AP_IoU_0.50_MaxDet_100', np.nan):0.3f} "
            f"FROC@0.1: {box_scores.get('mc_FROC_score_IoU_0.10', np.nan):0.3f} "
            f"FROC@0.5: {box_scores.get('mc_FROC_score_IoU_0.50', np.nan):0.3f} "
            f"FROC@0.1 (pool): {box_scores.get('FROC_score_IoU_0.10', np.nan):0.3f} "
        )

        # log own scores
        for key, item in box_scores.items():
            self.log(
                f"val/box_{key}",
                item,
                on_step=None,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        return metric_scores


class BoxWithRPNEvalMixin(BoxEvalMixin):
    def evaluation_init(self, plan: dict) -> Dict[str, AbstractEvaluator]:
        """
        Initialize `BoxWithRPNEvalMixin`
        This class extends the normal BoxEvalMixin with the evaluation
        of the Region Proposal Network. Class information from the RPN
        and the Ground Truth classes will be discarded for the evaluation
        of the RPN.

        Warnings:
        This `EvalMixin` only works with detection networks employing
        a `region proposal network(RPN)`! The evaluation of the RPN is performed
        additionally to the normal evaluation and thus this Mixin
        should not be combined with the `BoxEvalMixin`.

        Notes:
        make sure to call the super classes here!
        """
        evaluators = super().evaluation_init(plan=plan)
        if "rpn_boxes" in evaluators:
            raise RuntimeError("Found BoxWithRPNEvalMixin in evaluators, can not register a second one!")

        evaluators["rpn_boxes"] = BoxEvaluator.create(
            classes=["rpn_fg"],
            fast=True,
            save_dir=None,
        )
        return evaluators

    def evaluation_step(
        self,
        predictions: dict,
        targets: dict,
    ) -> None:
        """
        Evaluate a validation batch with metrics

        Args:
        predictions: dict with predictions.
        Exact keys depend on the module class
        targets: dict with ground truth.
        Exact keys depend on the module class.

        Notes:
        make sure to call the super classes here!
        """
        super().evaluation_step(predictions=predictions, targets=targets)

        pred_boxes = to_numpy(predictions["rpn_pred_boxes"])
        pred_scores = to_numpy(predictions["rpn_pred_scores"])
        pred_classes = to_numpy(predictions["rpn_pred_labels"])
        pred_classes_ones = [np.zeros_like(pc) for pc in pred_classes]

        gt_boxes = to_numpy(targets["target_boxes"])
        gt_classes = to_numpy(targets["target_classes"])
        gt_classes_ones = [np.zeros_like(gc) for gc in gt_classes]
        gt_ignore = None

        self.evaluators["rpn_boxes"].run_online_evaluation(
            pred_boxes=pred_boxes,
            pred_classes=pred_classes_ones,
            pred_scores=pred_scores,
            gt_boxes=gt_boxes,
            gt_classes=gt_classes_ones,
            gt_ignore=gt_ignore,
        )

    def evaluation_end(self) -> Dict[str, float]:
        """
        Compute validation metrics of epoch

        Notes:
        make sure to call the super classes here!
        """
        # collect other scores
        metric_scores = super().evaluation_end()

        # compute own scores
        rpn_scores, _ = self.evaluators["rpn_boxes"].finish_online_evaluation()
        self.evaluators["rpn_boxes"].reset()

        # add own scores
        metric_scores.update({f"rpn_{key}": item for key, item in rpn_scores.items()})

        # [optional] log own scores
        logger.info(
            "RPN Box::   "
            f"mAP@0.1:0.5:0.05: {rpn_scores['mAP_IoU_0.10_0.50_0.05_MaxDet_100']:0.3f}  "
            f"AP@0.1: {rpn_scores['AP_IoU_0.10_MaxDet_100']:0.3f} "
            f"AP@0.5: {rpn_scores['AP_IoU_0.50_MaxDet_100']:0.3f} "
            f"FROC@0.1: {rpn_scores['mc_FROC_score_IoU_0.10']:0.3f} "
            f"FROC@0.5: {rpn_scores['mc_FROC_score_IoU_0.50']:0.3f} "
        )

        # log own scores
        for key, item in rpn_scores.items():
            self.log(
                f"val_rpn/box_{key}",
                item,
                on_step=None,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        return metric_scores
