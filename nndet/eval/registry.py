# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import copy
from os import PathLike
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

from nndet.eval.case import CaseEvaluator
from nndet.eval.det import BoxEvaluator
from nndet.io.load import load_pickle
from nndet.utils.info import maybe_verbose_iterable


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
        :class:`nndet.eval.registry.BoxEvaluator`
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
        verbose=True,
        save_dir=save_dir,
        **kwargs,
    )
    logger.info(f"Created box evaluator: {evaluator}")

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
            case_ids=[case_id],
        )
    return evaluator.finish_online_evaluation()


def evaluate_box_dir_bootstrap(
    pred_dir: PathLike,
    gt_dir: PathLike,
    classes: Sequence[str],
    iterations: int = 1000,
    iqr: float = 0.95,
    seed: int = 0,
    **kwargs,
) -> Tuple[Dict, Dict]:
    """
    Run box evaluation inside a directory

    Args:
        pred_dir: path to dir with predictions
        gt_dir: path to dir with groud truth data
        classes: classes present in dataset
        iterations: number of bootstrapping iterations
        iqr: interquartile range
        kwargs: keyword arguments passed to box evaluator

    Returns:
        Dict[str, float]: dictionary with scalar values for evaluation
        Dict[str, np.ndarray]: dictionary with arrays, e.g. for visualization of graphs

    See Also:
        :class:`nndet.eval.registry.BoxEvaluator`

    Warning:
        This implementation uses a cached variant for the predictions
        and ground truth boxes. As such they need to fit into RAM, otherwise
        the script will crash.
    """
    rng = np.random.default_rng(seed=seed)

    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    case_ids = [p.stem.rsplit("_boxes", 1)[0] for p in pred_dir.iterdir() if p.is_file() and p.stem.endswith("_boxes")]
    logger.info(f"Found {len(case_ids)} for box evaluation in {pred_dir}")

    pred_cache = {}
    gt_cache = {}
    for case_id in case_ids:
        pred_cache[case_id] = load_pickle(pred_dir / f"{case_id}_boxes.pkl")
        _gt = np.load(str(gt_dir / f"{case_id}_boxes_gt.npz"), mmap_mode="r", allow_pickle=True)
        gt_cache[case_id] = {key: _gt[key] for key in _gt.keys()}

    res_scores = []
    res_curves = []
    logger.info(f"Running bootstrapping: iterations {iterations}, iqr {iqr}, seed {seed}")
    for bootstrap_idx in maybe_verbose_iterable(range(iterations)):
        case_id_idx = rng.integers(low=0, high=len(case_ids), size=len(case_ids))
        case_ids_bootstrap = [case_ids[i] for i in case_id_idx]
        _bootstrap_preds = [copy.deepcopy(pred_cache[cids]) for cids in case_ids_bootstrap]
        _bootstrap_gts = [copy.deepcopy(gt_cache[cids]) for cids in case_ids_bootstrap]

        evaluator = BoxEvaluator.create(
            classes=classes,
            fast=False,
            verbose=False,
            save_dir=None,
            do_criterion_eval=False,
            **kwargs,
        )
        if bootstrap_idx == 0:
            logger.info(f"Created box evaluator: {evaluator}")

        assert len(_bootstrap_preds) == len(_bootstrap_gts)
        for pred, gt in zip(_bootstrap_preds, _bootstrap_gts):
            evaluator.run_online_evaluation(
                pred_boxes=[pred["pred_boxes"]],
                pred_classes=[pred["pred_labels"]],
                pred_scores=[pred["pred_scores"]],
                gt_boxes=[gt["boxes"]],
                gt_classes=[gt["classes"]],
                gt_ignore=None,
                case_ids=None,
            )
        _scores, _curves = evaluator.finish_online_evaluation()
        _curves["__case_ids"] = case_ids_bootstrap
        _curves["__seed"] = seed
        res_scores.append(_scores)
        res_curves.append(_curves)

    res_scores_iqr = {}
    score_keys = list(res_scores[0].keys())
    for sk in score_keys:
        scores_array = np.array([x[sk] for x in res_scores])
        res_scores_iqr[sk] = {
            "mean": np.mean(scores_array),
            "median": np.median(scores_array),
            "iqr_low": np.quantile(scores_array, q=0.5 - iqr / 2),
            "iqr_high": np.quantile(scores_array, q=0.5 + iqr / 2),
            "std": np.std(scores_array),
            "__iqr": iqr,
            "__iterations": iterations,
            "__seed": seed,
        }
    return res_scores_iqr, res_scores, res_curves


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
        :class:`nndet.eval.registry.CaseEvaluator`
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
