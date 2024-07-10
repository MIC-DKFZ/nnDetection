import argparse
import itertools
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple, Union

import numpy as np
import torch
from loguru import logger

from nndet.core.boxes.nms import batched_nms
from nndet.core.boxes.wbc import batched_wbc
from nndet.eval.det.evaluator import BoxEvaluator
from nndet.io import load_pickle, save_pickle
from nndet.io.paths import get_task
from nndet.utils.check import env_guard
from nndet.utils.config import load_dataset_info
from nndet.utils.enums import EnsembleNMS
from nndet.utils.info import maybe_verbose_iterable


@env_guard
def entrypoint_ensemble_with_task():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument("fold", type=int, help="fold, use -1 for consolidated")
    parser.add_argument(
        "target_model",
        type=str,
        help="name of new model directory to save predictions in",
    )
    parser.add_argument(
        "source_models",
        type=str,
        nargs="+",
        help="models to ensemble, e.g. RetinaUNetV0_D3V001_3d",
    )
    parser.add_argument(
        "-i",
        "--iou",
        type=float,
        required=True,
        help="Define IoU threshold for ensembling",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        required=False,
        default="wbc",
        help="Define ensembling mode, one of 'wbc' or 'nms'.",
    )
    parser.add_argument(
        "--test",
        help="Test predictions. Otherwise prediction from validation folder are ensembled",
        action="store_true",
    )

    args = parser.parse_args()
    task = args.task
    fold = args.fold
    target_model = args.target_model
    source_models = args.source_models
    iou = args.iou
    mode = args.mode.lower()
    test = args.test

    if len(source_models) < 1:
        raise ValueError("Only a single model was passed. Ensembling can only be used for multiple models.")
    if mode not in ["wbc", "nms"]:
        raise ValueError(f"{mode} is not supported for mode, only 'nms' or 'wbc' are supported")
    mode = f"batched_{mode}"  # enums refer to batched modes

    if not (iou <= 1 and iou >= 0):
        raise ValueError("IoU need to be in the range [0, 1].")

    # env paths
    det_models = Path(os.getenv("det_models"))
    task = get_task(task, name=True, models=True)
    fold = "consolidated" if fold == -1 else f"fold{fold}"
    predictions_dir_name = "test_predictions" if test else "val_predictions"

    target_model_dir: Path = det_models / task / target_model / fold
    target_model_dir.mkdir(parents=True, exist_ok=True)
    target_prediction_dir = target_model_dir / predictions_dir_name

    source_model_dirs: List[Path] = [det_models / task / m / fold for m in source_models]
    for smd in source_model_dirs:
        if not smd.is_dir():
            raise ValueError(f"Expected {smd} to be a model dir but this directory does not exist.")
    source_prediction_dirs: List[Path] = [smd / predictions_dir_name for smd in source_model_dirs]
    for spd in source_prediction_dirs:
        if not spd.is_dir():
            raise ValueError(f"Expected {spd} to be a prediction dir but this directory does not exist.")

    _ensemble(
        target_prediction_dir=target_prediction_dir,
        source_prediction_dirs=source_prediction_dirs,
        mode=mode,
        iou=iou,
    )


@env_guard
def entrypoint_ensemble_with_models():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target_model_path",
        type=str,
        help="path to new model directory",
    )
    parser.add_argument(
        "source_model_paths",
        type=str,
        nargs="+",
        help="path to models to ensemble, e.g. RetinaUNetV0_D3V001_3d",
    )
    parser.add_argument(
        "-i",
        "--iou",
        type=float,
        required=True,
        help="Define IoU threshold for ensembling",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        required=False,
        default="wbc",
        help="Define ensembling mode, one of 'wbc' or 'nms'.",
    )
    parser.add_argument(
        "--test",
        help=("Test predictions. Otherwise prediction from validation folder are ensembled"),
        action="store_true",
    )

    args = parser.parse_args()
    target_model_path = args.target_model_path
    source_model_paths = args.source_model_paths
    iou = args.iou
    mode = args.mode.lower()
    test = args.test

    if len(source_model_paths) < 1:
        raise ValueError("Only a single model was passed. Ensembling can only be used for multiple models.")
    if mode not in ["wbc", "nms"]:
        raise ValueError(f"{mode} is not supported for mode, only 'nms' or 'wbc' are supported")
    mode = f"batched_{mode}"  # enums refer to batched modes

    if not (iou <= 1 and iou >= 0):
        raise ValueError("IoU need to be in the range [0, 1].")

    # env paths
    predictions_dir_name = "test_predictions" if test else "val_predictions"

    target_model_dir: Path = Path(target_model_path)
    target_model_dir.mkdir(parents=True, exist_ok=True)
    target_prediction_dir = target_model_dir / predictions_dir_name

    source_model_dirs: List[Path] = [Path(smp) for smp in source_model_paths]
    for smd in source_model_dirs:
        if not smd.is_dir():
            raise ValueError(f"Expected {smd} to be a model dir but this directory does not exist.")
    source_prediction_dirs: List[Path] = [smd / predictions_dir_name for smd in source_model_dirs]
    for spd in source_prediction_dirs:
        if not spd.is_dir():
            raise ValueError(f"Expected {spd} to be a prediction dir but this directory does not exist.")

    _ensemble(
        target_prediction_dir=target_prediction_dir,
        source_prediction_dirs=source_prediction_dirs,
        mode=mode,
        iou=iou,
    )


@env_guard
def entrypoint_ensemble_with_folders():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target_prediction_path",
        type=str,
        help="path to directory where new predictions should be saved",
    )
    parser.add_argument(
        "source_prediction_paths",
        type=str,
        nargs="+",
        help="paths to directories with predictions",
    )
    parser.add_argument(
        "-i",
        "--iou",
        type=float,
        required=True,
        help="Define IoU threshold for ensembling",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        required=False,
        default="wbc",
        help="Define ensembling mode, one of 'wbc' or 'nms'.",
    )

    args = parser.parse_args()
    target_prediction_path = args.target_prediction_path
    source_prediction_paths = args.source_prediction_paths
    iou = args.iou
    mode = args.mode.lower()

    if len(source_prediction_paths) < 1:
        raise ValueError("Only a single model was passed. Ensembling can only be used for multiple models.")
    if mode not in ["wbc", "nms"]:
        raise ValueError(f"{mode} is not supported for mode, only 'nms' or 'wbc' are supported")
    mode = f"batched_{mode}"  # enums refer to batched modes

    if not (iou <= 1 and iou >= 0):
        raise ValueError("IoU need to be in the range [0, 1].")

    target_prediction_dir = Path(target_prediction_path)
    target_prediction_dir.mkdir(parents=True, exist_ok=True)

    source_prediction_dirs: List[Path] = [Path(spp) for spp in source_prediction_paths]
    for spd in source_prediction_dirs:
        if not spd.is_dir():
            raise ValueError(f"Expected {spd} to be a prediction dir but this directory does not exist.")

    _ensemble(
        target_prediction_dir=target_prediction_dir,
        source_prediction_dirs=source_prediction_dirs,
        mode=mode,
        iou=iou,
    )


def _get_case_ids(source_prediction_dirs: List[Path]) -> List[str]:
    """
    Retrieve case ids from directories.

    Args:
        source_prediction_dirs: paths to directories with predictions

    Returns:
        List[str]: case ids

    Raises:
        ValueError: raised if different cases are present in the prediction
            directories.
    """
    case_ids: List[List[str]] = [
        [pp.name.rsplit("_", 1)[0] for pp in pd.glob("*_boxes.pkl")] for pd in source_prediction_dirs
    ]
    case_ids: List[Set[str]] = [set(ci) for ci in case_ids]

    if not all([case_ids[0] == case_ids[i + 1] for i in range(len(source_prediction_dirs) - 1)]):
        raise ValueError("Found different case ids in prediction directories.")
    return list(case_ids[0])


@env_guard
def _ensemble(
    target_prediction_dir: Path,
    source_prediction_dirs: List[Path],
    mode: Union[str, EnsembleNMS],
    iou: float,
) -> None:
    case_ids = _get_case_ids(source_prediction_dirs)
    target_prediction_dir.mkdir(exist_ok=True)
    mode = EnsembleNMS(mode)

    for cid in maybe_verbose_iterable(case_ids):
        boxes = []
        scores = []
        labels = []
        for pd in source_prediction_dirs:
            pred = load_pickle(pd / f"{cid}_boxes.pkl")
            boxes.append(pred["pred_boxes"])
            scores.append(pred["pred_scores"])
            labels.append(pred["pred_labels"])

        boxes = np.concatenate(boxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        if mode == EnsembleNMS.NMS:
            pred_boxes, pred_scores, pred_labels, _ = batched_nms(
                boxes=torch.from_numpy(boxes),
                scores=torch.from_numpy(scores),
                labels=torch.from_numpy(labels),
                iou_thresh=iou,
            )
        elif mode == EnsembleNMS.WBC:
            pred_boxes, pred_scores, pred_labels, _ = batched_wbc(
                boxes=torch.from_numpy(boxes),
                scores=torch.from_numpy(scores),
                labels=torch.from_numpy(labels),
                weights=torch.from_numpy(np.ones_like(scores)),
                iou_thresh=iou,
                n_exp_preds=torch.from_numpy(np.ones_like(scores) * len(source_prediction_dirs)),
                use_area=False,
                missing_weight=1.0,
            )

        pred_boxes = pred_boxes.cpu().numpy()
        pred_scores = pred_scores.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()

        pred_ensemble = {
            "pred_boxes": pred_boxes,
            "pred_scores": pred_scores,
            "pred_labels": pred_labels,
        }
        save_pickle(pred_ensemble, target_prediction_dir / f"{cid}_boxes.pkl")


@env_guard
def entrypoint_determine_best_ensemble_with_task():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument(
        "new_model",
        type=str,
        help="name of new model directory to save predictions in",
        # default="nnDetectonV2_ensemble",
    )
    parser.add_argument(
        "models",
        type=str,
        nargs="+",
        help="models to ensemble",
    )

    args = parser.parse_args()
    task: str = args.task
    models: List[str] = args.models
    new_model: str = args.new_model

    # prepare paths
    task = get_task(task, name=True, models=True)
    data_dir_task = Path(os.getenv("det_data")) / task
    data_cfg = load_dataset_info(data_dir_task)

    new_model_dir = Path(os.getenv("det_models")) / task / new_model
    new_model_dir.mkdir(exist_ok=True)

    prediction_dirs = {
        model: Path(os.getenv("det_models")) / task / model / "consolidated" / "val_predictions" for model in models
    }
    for model, pd in prediction_dirs.items():
        if not pd.is_dir():
            raise ValueError(f"Prediction directory {pd} for model {model} does not exist.")

    # logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<level>{level} {message}</level>",
        level="INFO",
        colorize=True,
    )
    logger.add(new_model_dir / "determine_ensemble.log", level="INFO")
    current_time = datetime.now()
    current_time_str = current_time.strftime("%d/%m/%Y %H:%M:%S")
    logger.info(f"+++ Running 'find best ensemble and paramter' {current_time_str} +++")
    logger.info(f"Looking for ensemble models out of {models}")

    best_model_subset, best_iou, _ = _determine_best_ensemble_and_parameter(
        prediction_dirs=prediction_dirs,
        ground_truth_dir=data_dir_task / "preprocessed" / "labelsTr",
        classes=list(data_cfg["labels"].values()),
        optim_metric="AP_IoU_0.10",
    )

    # print instructions for ensembling
    if len(best_model_subset) == 1:
        logger.info(
            "A single model was determined to be the best. No ensembling required. Follow the "
            "normal instructions to run validation and inference."
        )
    else:
        logger.info("A model ensemble was determined to be the best.")
        logger.info("To obtain inference results, follow the instructions below.")
        for model in best_model_subset:
            logger.info(
                f"Run inference for model {model} via: `nndet_predict_with_task [data] "
                f"$det_models/{task}/{model}/consolidated/test_predictions {task} {model} -1`"
            )
        logger.info(
            f"Than run: `nndet_ensemble_with_task {task} -1 {new_model} "
            f"{' '.join(best_model_subset)} -i {best_iou} -m wbc [--test]` to ensemble the model "
            "predictions."
        )


def _determine_best_ensemble_and_parameter(
    prediction_dirs: Dict[str, os.PathLike],
    ground_truth_dir: os.PathLike,
    classes: List[str],
    optim_metric: str = "AP_IoU_0.10",
) -> Tuple[List[str], float, float]:
    """
    Determines the best IoU for WBC to ensemble predictions from different
    models

    Args:
        prediction_dirs: dictionary with model names as keys and paths to
            prediction directories as values
        ground_truth_dir: path to directory containing ground truth
            bounding boxes
        classes: classes present in dataset
        optim_metric: metric to optimize. Defaults to "AP_IoU_0.10".

    Returns:
        Tuple[List[str], float, float]: return the best model subset, the best
            IoU and the best optimization metric value
    """
    prediction_dirs = {k: Path(v) for k, v in prediction_dirs.items()}
    ground_truth_dir = Path(ground_truth_dir)

    # define ensembling setup

    # iou_values = np.arange(0.0, 1.05, 0.05)
    iou_values = np.arange(0.0, 0.55, 0.05)
    iou_values[0] = 1e-5  # use small value to indicate any overlap, 0 not working

    # determine predictions
    case_ids = _get_case_ids(list(prediction_dirs.values()))

    # determine all possible ensemble configurations
    model_names = prediction_dirs.keys()
    all_subsets = list(
        itertools.chain.from_iterable(itertools.combinations(model_names, r) for r in range(len(model_names) + 1))
    )
    assert all_subsets[0] == ()
    all_subsets = all_subsets[1:]  # remove empty subset

    # sweep
    best_optim_metric = 0
    best_iou = None
    best_model_subset = None
    logger.info("Start sweeping through all possible ensemble configurations.")
    for model_subset in all_subsets:  # iterate all model combinations
        for iou_idx, iou in enumerate(iou_values):  # iterate all iou values
            if len(model_subset) == 1 and iou_idx > 0:
                continue  # skip iou optimization for single model

            evaluator = BoxEvaluator.create(
                classes=classes,
                fast=True,
                verbose=False,
                save_dir=None,
            )

            # load & ensemble predictions
            case_predictions = _load_ensemble_predictions([prediction_dirs[m] for m in model_subset], case_ids, iou)

            # evaluate
            for cid in maybe_verbose_iterable(case_ids):
                # eval
                gt = np.load(str(ground_truth_dir / f"{cid}_boxes_gt.npz"), allow_pickle=True)
                evaluator.run_online_evaluation(
                    pred_boxes=[case_predictions[cid]["pred_boxes"]],
                    pred_classes=[case_predictions[cid]["pred_labels"]],
                    pred_scores=[case_predictions[cid]["pred_scores"]],
                    gt_boxes=[gt["boxes"]],
                    gt_classes=[gt["classes"]],
                    gt_ignore=None,
                    case_ids=[cid],
                )

            # store results
            eval_scores, _ = evaluator.finish_online_evaluation()
            logger.info(f"Evaluated ensemble {model_subset} with IoU {iou}: {eval_scores[optim_metric]}")
            if eval_scores[optim_metric] > best_optim_metric:
                logger.info("New best ensemble configuration found.")
                best_optim_metric = eval_scores[optim_metric]
                best_iou = iou
                best_model_subset = model_subset

    assert best_iou is not None
    assert best_model_subset is not None

    logger.info(
        f"Best ensemble configuration: {best_model_subset} with IoU {best_iou} and "
        f"{optim_metric} {best_optim_metric}"
    )
    return best_model_subset, best_iou, best_optim_metric


def _load_ensemble_predictions(
    prediction_dirs: List[os.PathLike],
    case_ids: Sequence[str],
    iou: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Helper function to load and ensemble predictions from different models

    Args:
        prediction_dirs: sequence of directories containing predictions from
            different models
        case_ids: case ids which should be used for ensembling
        iou: the iou thresold used to determined if predictins should be
            grouped

    Returns:
        Dict[str, Dict[str, np.ndarray]]: ensembled predictions
    """
    ensemble_function = batched_wbc
    case_predictions = {}
    num_models = len(prediction_dirs)

    for cid in maybe_verbose_iterable(case_ids):
        if len(prediction_dirs) == 1:
            pred = load_pickle(prediction_dirs[0] / f"{cid}_boxes.pkl")
            case_predictions[cid] = pred
        else:
            boxes = []
            scores = []
            labels = []
            for prediction_dir in prediction_dirs:
                pred = load_pickle(prediction_dir / f"{cid}_boxes.pkl")
                boxes.append(pred["pred_boxes"])
                scores.append(pred["pred_scores"])
                labels.append(pred["pred_labels"])

            boxes = np.concatenate(boxes, axis=0)
            scores = np.concatenate(scores, axis=0)
            labels = np.concatenate(labels, axis=0)
            pred_boxes, pred_scores, pred_labels, _ = ensemble_function(
                boxes=torch.from_numpy(boxes),
                scores=torch.from_numpy(scores),
                labels=torch.from_numpy(labels),
                weights=torch.from_numpy(np.ones_like(scores)),
                iou_thresh=iou,
                n_exp_preds=torch.from_numpy(np.ones_like(scores) * num_models),
                use_area=False,
                missing_weight=1.0,
            )
            case_predictions[cid] = {
                "pred_boxes": pred_boxes.cpu().numpy(),
                "pred_scores": pred_scores.cpu().numpy(),
                "pred_labels": pred_labels.cpu().numpy(),
            }
    return case_predictions


if __name__ == "__main__":
    entrypoint_determine_best_ensemble_with_task()
