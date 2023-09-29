import argparse
import os
from pathlib import Path
from typing import List, Set, Union

import numpy as np
import torch

from nndet.core.boxes.nms import batched_nms
from nndet.core.boxes.wbc import batched_wbc
from nndet.io import load_pickle, save_pickle
from nndet.io.paths import get_task
from nndet.utils.check import env_guard
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
        help="Test predictions. Otherwise prediction from validation folder are ensembled",
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


if __name__ == "__main__":
    entrypoint_ensemble_with_task()
