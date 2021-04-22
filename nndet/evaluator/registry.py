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

from os import PathLike
from pathlib import Path
from typing import Dict, Sequence, Optional, Tuple

import numpy as np
from loguru import logger

from nndet.io.load import load_pickle, save_json, save_pickle
from nndet.evaluator.det import BoxEvaluator
from nndet.evaluator.case import CaseEvaluator
from nndet.evaluator.seg import PerCaseSegmentationEvaluator


def save_metric_output(scores, curves, base_dir, name):
    """
    Helper function to save output of the function in a nice format
    """
    scores_string = {str(key): str(item) for key, item in scores.items()}
    
    save_json(scores_string, base_dir / f"{name}.json")
    save_pickle({"scores": scores, "curves": curves}, base_dir / f"{name}.pkl")


def evaluate_box_dir(
    pred_dir: PathLike,
    gt_dir: PathLike,
    classes: Sequence[str],
    save_dir: Optional[Path] = None,
    ) -> Tuple[Dict, Dict]:
    """
    Run box evaluation inside a directory

    Args:
        pred_dir: path to dir with predictions
        gt_dir: path to dir with groud truth data
        classes: classes present in dataset
        save_dir: optional path to save plots

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
    case_ids = [p.stem.rsplit('_boxes', 1)[0] for p in pred_dir.iterdir()
                if p.is_file() and p.stem.endswith("_boxes")]
    logger.info(f"Found {len(case_ids)} for box evaluation in {pred_dir}")

    evaluator = BoxEvaluator.create(classes=classes,
                                    fast=False,
                                    verbose=False,
                                    save_dir=save_dir,
                                    )

    for case_id in case_ids:
        gt = np.load(str(gt_dir / f"{case_id}_boxes_gt.npz"), allow_pickle=True)
        pred = load_pickle(pred_dir / f"{case_id}_boxes.pkl")
        evaluator.run_online_evaluation(
            pred_boxes=[pred["pred_boxes"]], pred_classes=[pred["pred_labels"]],
            pred_scores=[pred["pred_scores"]], gt_boxes=[gt["boxes"]],
            gt_classes=[gt["classes"]], gt_ignore=None,
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
    case_ids = [p.stem.rsplit('_boxes', 1)[0] for p in pred_dir.iterdir()
                if p.is_file() and p.stem.endswith("_boxes")]
    logger.info(f"Found {len(case_ids)} for case evaluation in {pred_dir}")

    evaluator = CaseEvaluator.create(classes=classes,
                                     target_class=target_class,
                                     )

    for case_id in case_ids:
        gt = np.load(str(gt_dir / f"{case_id}_boxes_gt.npz"), allow_pickle=True)
        pred = load_pickle(pred_dir / f"{case_id}_boxes.pkl")
        evaluator.run_online_evaluation(
            pred_classes=[pred["pred_labels"]],
            pred_scores=[pred["pred_scores"]],
            gt_classes=[gt["classes"]]
            )
    return evaluator.finish_online_evaluation()


def evaluate_seg_dir(
    pred_dir: PathLike,
    gt_dir: PathLike,
    classes: Sequence[str],
    ) -> Tuple[Dict, None]:
    """
    Compute dice metric across a directory

    Args:
        pred_dir: path to dir with predictions
        gt_dir: path to dir with groud truth data
        classes: classes present in dataset

    Returns:
        Dict[str, float]: dictionary with scalar values for evaluation
        None

    See Also:
        :class:`nndet.evaluator.registry.PerCaseSegmentationEvaluator`
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    case_ids = [p.stem.rsplit('_seg', 1)[0] for p in pred_dir.iterdir()
                if p.is_file() and p.stem.endswith("_seg")]
    logger.info(f"Found {len(case_ids)} for seg evaluation in {pred_dir}")

    evaluator = PerCaseSegmentationEvaluator.create(classes=classes)

    for case_id in case_ids:
        gt = np.load(str(gt_dir / f"{case_id}_seg_gt.npz"), allow_pickle=True)["seg"] # 1, dims
        pred = load_pickle(pred_dir / f"{case_id}_seg.pkl")
        evaluator.run_online_evaluation(
            seg=pred[None],
            target=gt,
            )
    return evaluator.finish_online_evaluation()
