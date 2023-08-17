# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from os import PathLike
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

from nndet.evaluator.case import CaseEvaluator
from nndet.evaluator.det import BoxEvaluator
from nndet.io.load import load_pickle


def evaluate_box_dir(
    pred_dir: PathLike,
    gt_dir: PathLike,
    classes: Sequence[str],
    save_dir: Optional[Path] = None,
    **kwargs,
) -> Tuple[Dict, Dict]:
    """
    Run box evaluation inside a directory

    Args:
        pred_dir: path to dir with predictions
        gt_dir: path to dir with groud truth data
        classes: classes present in dataset
        save_dir: optional path to save plots
        kwargs: keyword arguments passed to box evaluator

    Returns:
        Dict[str, float]: dictionary with scalar values for evaluation
        Dict[str, np.ndarray]: dictionary with arrays, e.g. for visualization of graphs

    See Also:
        :class:`nndet.evaluator.registry.BoxEvaluator`
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    case_ids = [p.stem.rsplit("_boxes", 1)[0] for p in pred_dir.iterdir() if p.is_file() and p.stem.endswith("_boxes")]
    logger.info(f"Found {len(case_ids)} for box evaluation in {pred_dir}")

    evaluator = BoxEvaluator.create(
        classes=classes,
        fast=False,
        verbose=False,
        save_dir=save_dir,
        **kwargs,
    )

    for case_id in case_ids:
        gt = np.load(str(gt_dir / f"{case_id}_boxes_gt.npz"), allow_pickle=True)
        pred = load_pickle(pred_dir / f"{case_id}_boxes.pkl")

        evaluator.run_online_evaluation(
            pred_boxes=[pred["pred_boxes"]],
            pred_classes=[pred["pred_labels"]],
            pred_scores=[pred["pred_scores"]],
            gt_boxes=[gt["boxes"]],
            gt_classes=[gt["classes"]],
            gt_ignore=None,
            case_id=case_id,
        )
    return evaluator.finish_online_evaluation()


def evaluate_case_dir(
    pred_dir: PathLike,
    gt_dir: PathLike,
    classes: Sequence[str],
    target_class: Optional[int] = None,
) -> Tuple[Dict, Dict]:
    """
    Run evaluation of case results inside a directory

    Args:
        pred_dir: path to dir with predictions
        gt_dir: path to dir with groud truth data
        classes: classes present in dataset
        target_class in case of multiple classes, specify a target class
            to evaluate in a target class vs rest setting

    Returns:
        Dict[str, float]: dictionary with scalar values for evaluation
        Dict[str, np.ndarray]: dictionary with arrays, e.g. for visualization of graph)

    See Also:
        :class:`nndet.evaluator.registry.CaseEvaluator`
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    case_ids = [p.stem.rsplit("_boxes", 1)[0] for p in pred_dir.iterdir() if p.is_file() and p.stem.endswith("_boxes")]
    logger.info(f"Found {len(case_ids)} for case evaluation in {pred_dir}")

    evaluator = CaseEvaluator.create(
        classes=classes,
        target_class=target_class,
    )

    for case_id in case_ids:
        gt = np.load(str(gt_dir / f"{case_id}_boxes_gt.npz"), allow_pickle=True)
        pred = load_pickle(pred_dir / f"{case_id}_boxes.pkl")
        evaluator.run_online_evaluation(
            pred_classes=[pred["pred_labels"]],
            pred_scores=[pred["pred_scores"]],
            gt_classes=[gt["classes"]],
        )
    return evaluator.finish_online_evaluation()
