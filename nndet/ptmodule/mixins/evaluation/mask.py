# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

import torch
from loguru import logger

import nndet.core.ops_torch as ops_torch
from nndet.evaluator import AbstractEvaluator
from nndet.evaluator.det import MaskEvaluator
from nndet.ptmodule.mixins.evaluation.base import EvalMixin
from nndet.utils.tensor import to_numpy
from nndet.utils.typing import ND_TUPLE_INT


class ScoreMasksEvalMixin(EvalMixin):
    def evaluation_init(self, plan: dict) -> Dict[str, AbstractEvaluator]:
        """
        Initialize `MaskEvaluator`
        Masks are resized with nearest neighbor and an cutoff value of 0.5 .

        Notes:
        make sure to call the super classes here!
        """
        evaluators = super().evaluation_init(plan=plan)
        if "score_masks" in evaluators:
            raise RuntimeError("Found ScoreMasksEvaluator in evaluators, can not register a second one!")

        _classes = [f"class{c}" for c in range(plan["architecture"]["classifier_classes"])]
        evaluators["score_masks"] = MaskEvaluator.create(
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

        _pred_boxes: List[torch.Tensor] = predictions["pred_boxes"]
        _pred_masks_probs: List[torch.Tensor] = predictions["pred_masks"]
        _image_spatial_size: ND_TUPLE_INT = predictions["__pred_image_spatial_size"]
        target_binary_masks: List[torch.Tensor] = targets["target_binary_masks"]
        assert len(_pred_masks_probs) == len(target_binary_masks) == len(_pred_boxes)
        pred_masks = []
        for idx in range(len(target_binary_masks)):
            # breakpoint()
            pred_bin_masks = ops_torch.roi_mask_to_image_mask(
                boxes=_pred_boxes[idx],
                masks=_pred_masks_probs[idx],
                image_shape=_image_spatial_size,
                threshold=0.5,
                mode="nearest",
            )
            assert pred_bin_masks.ndim == len(_image_spatial_size) + 1
            pred_masks.append(pred_bin_masks)

        pred_masks = to_numpy(pred_masks)
        pred_classes = to_numpy(predictions["pred_mask_labels"])
        pred_scores = to_numpy(predictions["pred_mask_scores"])

        gt_masks = to_numpy(target_binary_masks)
        gt_classes = to_numpy(targets["target_classes"])
        gt_ignore = None

        self.evaluators["score_masks"].run_online_evaluation(
            pred_boxes=pred_masks,
            pred_classes=pred_classes,
            pred_scores=pred_scores,
            gt_boxes=gt_masks,
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
        box_scores, _ = self.evaluators["score_masks"].finish_online_evaluation()
        self.evaluators["score_masks"].reset()

        # add own scores
        metric_scores.update({f"mask_{k}": i for k, i in box_scores.items()})

        # [optional] log own scores
        logger.info(
            "Mask::   "
            f"mAP@0.1:0.5:0.05: {box_scores['mAP_IoU_0.10_0.50_0.05_MaxDet_100']:0.3f}  "
            f"AP@0.1: {box_scores['AP_IoU_0.10_MaxDet_100']:0.3f}  "
            f"AP@0.5: {box_scores['AP_IoU_0.50_MaxDet_100']:0.3f} "
            f"FROC@0.1: {box_scores['mc_FROC_score_IoU_0.10']:0.3f} "
            f"FROC@0.5: {box_scores['mc_FROC_score_IoU_0.50']:0.3f} "
            f"FROC@0.1 (pool): {box_scores['FROC_score_IoU_0.10']:0.3f} "
        )

        # log own scores
        for key, item in box_scores.items():
            self.log(
                f"val_mask/mask_{key}",
                item,
                on_step=None,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        return metric_scores
