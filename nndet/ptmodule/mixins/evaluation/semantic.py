# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

from loguru import logger

from nndet.evaluator import AbstractEvaluator
from nndet.evaluator.seg import SegmentationEvaluator
from nndet.ptmodule.mixins.evaluation.base import EvalMixin
from nndet.utils.tensor import to_numpy


class SemanticEvalMixin(EvalMixin):
    """
    This Mixin only works with BoxMixin!
    BoxMixin needs to be subclassed last e.g.
    `Module(.. SemanticMixin, BoxMixin, ..)`
    """

    def evaluation_init(self, plan: dict) -> Dict[str, AbstractEvaluator]:
        """
        Initialize SegmentationEvaluator

        Notes:
        make sure to call the super classes here!
        """
        evaluators = super().evaluation_init(plan=plan)
        if "semantic" in evaluators:
            raise RuntimeError("Found SegmentationEvaluator in evaluators, can not register a second one!")

        evaluators["semantic"] = SegmentationEvaluator.create()
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

        pred_seg = to_numpy(predictions["pred_seg"])
        gt_seg = to_numpy(targets["target_seg"])

        self.evaluators["semantic"].run_online_evaluation(
            seg_probs=pred_seg,
            target=gt_seg,
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
        seg_scores, _ = self.evaluators["semantic"].finish_online_evaluation()
        self.evaluators["semantic"].reset()

        # add own scores
        metric_scores.update(seg_scores)

        # [optional] log own scores
        logger.info(f"SS::   (Prox) Dice: {seg_scores['seg_dice']:0.3f}")

        # log own scores
        for key, item in seg_scores.items():
            self.log(
                f"val_seg/{key}",
                item,
                on_step=None,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        return metric_scores


class SemanticFgEvalMixin(EvalMixin):
    """
    Run segmentation evaluation in FG mode

    This Mixin only works with BoxMixin!
    BoxMixin needs to be subclassed last e.g.
    `Module(.. SemanticMixin, BoxMixin, ..)`
    """

    def evaluation_init(self, plan: dict) -> Dict[str, AbstractEvaluator]:
        """
        Initialize SegmentationEvaluator

        Notes:
        make sure to call the super classes here!
        """
        evaluators = super().evaluation_init(plan=plan)
        if "semantic_fg" in evaluators:
            raise RuntimeError("Found SegmentationEvaluator in evaluators, can not register a second one!")

        evaluators["semantic_fg"] = SegmentationEvaluator.create(fg_mode=True)
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

        pred_seg = to_numpy(predictions["pred_seg"])
        gt_seg = to_numpy(targets["target_seg"])

        self.evaluators["semantic_fg"].run_online_evaluation(
            seg_probs=pred_seg,
            target=gt_seg,
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
        seg_scores, _ = self.evaluators["semantic_fg"].finish_online_evaluation()
        self.evaluators["semantic_fg"].reset()

        # add own scores
        metric_scores.update(seg_scores)

        # [optional] log own scores
        logger.info(f"SS FG::   (Prox) Dice: {seg_scores['seg_dice']:0.3f}")

        # log own scores
        for key, item in seg_scores.items():
            self.log(
                f"val_seg/{key}",
                item,
                on_step=None,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        return metric_scores
