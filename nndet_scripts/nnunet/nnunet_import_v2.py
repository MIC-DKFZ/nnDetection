# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import shutil
import sys
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path, PurePath
from typing import Optional, Sequence

import numpy as np
from hydra import initialize_config_module
from loguru import logger

from nndet.eval.registry import evaluate_box_dir
from nndet.io import get_task, load_json, load_pickle, save_pickle
from nndet.io.load import save_json
from nndet.utils.clustering import softmax_to_instances
from nndet.utils.config import compose
from nndet.utils.info import maybe_verbose_iterable

TARGET_METRIC = "mAP_IoU_0.10_0.50_0.05"


"""
nnU-Net V2 reverts all preprocessing operatins for its probabilities output as well
So we can skip some steps here.
"""

def import_nnunet_boxes(
    # settings
    nnunet_prediction_dir: os.PathLike,
    save_dir: os.PathLike,
    boxes_gt_dir: os.PathLike,
    classes: Sequence[str],
    stuff: Optional[Sequence[int]] = None,
    num_workers: int = 6,
):
    assert nnunet_prediction_dir.is_dir(), f"{nnunet_prediction_dir} is not a dir"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    # create sweep dir
    sweep_dir = Path(nnunet_prediction_dir)
    postprocessing_settings = {}

    # optimize import method
    logger.info("Looking for optimal clustering mode")
    modes = ["voted", "connected"]
    scores = []
    for mode in modes:
        # create temp dir
        sweep_prediction = sweep_dir / f"sweep_mode_{mode}"
        sweep_prediction.mkdir(parents=True)

        # import with settings
        import_dir(
            nnunet_prediction_dir=nnunet_prediction_dir,
            target_dir=sweep_prediction,
            mode=mode,
            save_seg=False,
            save_iseg=False,
            stuff=stuff,
            num_workers=num_workers,
        )

        # evaluate
        _scores, _ = evaluate_box_dir(
            pred_dir=sweep_prediction,
            gt_dir=boxes_gt_dir,
            classes=classes,
            save_dir=None,
        )
        scores.append(_scores[TARGET_METRIC])
        summary.append({f"Mode {mode}": _scores[TARGET_METRIC]})
        logger.info(f"Mode {mode} :: {_scores[TARGET_METRIC]}")
        shutil.rmtree(sweep_prediction)

    idx = int(np.argmax(scores))
    postprocessing_settings["mode"] = modes[idx]
    logger.info(f"Found mode {modes[idx]} with score {scores[idx]}")

    # optimize min num voxels
    logger.info("Looking for optimal min voxel size")
    min_num_voxel_settings = [0, 5, 10, 15, 20]
    scores = []
    for min_num_voxel in min_num_voxel_settings:
        # create temp dir
        sweep_prediction = sweep_dir / f"sweep_min_voxel{min_num_voxel}"
        sweep_prediction.mkdir(parents=True)

        # import with settings
        import_dir(
            nnunet_prediction_dir=nnunet_prediction_dir,
            target_dir=sweep_prediction,
            min_num_voxel=min_num_voxel,
            save_seg=False,
            save_iseg=False,
            stuff=stuff,
            num_workers=num_workers,
        )

        # evaluate
        _scores, _ = evaluate_box_dir(
            pred_dir=sweep_prediction,
            gt_dir=boxes_gt_dir,
            classes=classes,
            save_dir=None,
        )
        scores.append(_scores[TARGET_METRIC])
        summary.append({f"Min voxel {min_num_voxel}": _scores[TARGET_METRIC]})
        logger.info(f"Min voxel {min_num_voxel} :: {_scores[TARGET_METRIC]}")
        shutil.rmtree(sweep_prediction)

    idx = int(np.argmax(scores))
    postprocessing_settings["min_num_voxel"] = min_num_voxel_settings[idx]
    logger.info(f"Found min num voxel {min_num_voxel_settings[idx]} with score {scores[idx]}")

    # optimize score threshold
    logger.info("Looking for optimal min probability threshold")
    min_threshold_settings = [None, 0.1, 0.2, 0.3, 0.4, 0.5]
    scores = []
    for min_threshold in min_threshold_settings:
        # create temp dir
        sweep_prediction = sweep_dir / f"sweep_min_threshold_{min_threshold}"
        sweep_prediction.mkdir(parents=True)

        # import with settings
        import_dir(
            nnunet_prediction_dir=nnunet_prediction_dir,
            target_dir=sweep_prediction,
            min_threshold=min_threshold,
            save_seg=False,
            save_iseg=False,
            stuff=stuff,
            num_workers=num_workers,
            **postprocessing_settings,
        )

        # evaluate
        _scores, _ = evaluate_box_dir(
            pred_dir=sweep_prediction,
            gt_dir=boxes_gt_dir,
            classes=classes,
            save_dir=None,
        )
        scores.append(_scores[TARGET_METRIC])
        summary.append({f"Min score {min_threshold}": _scores[TARGET_METRIC]})
        logger.info(f"Min score {min_threshold} :: {_scores[TARGET_METRIC]}")
        shutil.rmtree(sweep_prediction)

    idx = int(np.argmax(scores))
    postprocessing_settings["min_threshold"] = min_threshold_settings[idx]
    logger.info(f"Found min threshold {min_threshold_settings[idx]} with score {scores[idx]}")

    logger.info("Looking for best probability aggregation")
    aggreagtion_settings = ["max", "median", "mean", "percentile95"]
    scores = []
    for aggregation in aggreagtion_settings:
        # create temp dir
        sweep_prediction = sweep_dir / f"sweep_aggregation_{aggregation}"
        sweep_prediction.mkdir(parents=True)

        # import with settings
        import_dir(
            nnunet_prediction_dir=nnunet_prediction_dir,
            target_dir=sweep_prediction,
            aggregation=aggregation,
            save_seg=False,
            save_iseg=False,
            stuff=stuff,
            num_workers=num_workers,
            **postprocessing_settings,
        )
        # evaluate
        _scores, _ = evaluate_box_dir(
            pred_dir=sweep_prediction,
            gt_dir=boxes_gt_dir,
            classes=classes,
            save_dir=None,
        )
        scores.append(_scores[TARGET_METRIC])
        summary.append({f"Aggreagtion {aggregation}": _scores[TARGET_METRIC]})
        logger.info(f"Aggreagtion {aggregation} :: {_scores[TARGET_METRIC]}")
        shutil.rmtree(sweep_prediction)

    idx = int(np.argmax(scores))
    postprocessing_settings["aggregation"] = aggreagtion_settings[idx]
    logger.info(f"Found aggregation {aggreagtion_settings[idx]} with score {scores[idx]}")

    save_pickle(postprocessing_settings, save_dir / "postprocessing.pkl")
    save_json(summary, save_dir / "summary.json")
    return postprocessing_settings


def import_dir(
    nnunet_prediction_dir: os.PathLike,
    target_dir: Optional[os.PathLike] = None,
    aggregation="max",
    min_num_voxel=0,
    min_threshold=None,
    mode: str = "voted",
    save_seg: bool = True,
    save_iseg: bool = True,
    stuff: Optional[Sequence[int]] = None,
    num_workers: int = 6,
):
    source = [f for f in nnunet_prediction_dir.iterdir() if f.suffix == ".npz"]

    _fn = partial(
        import_single_case,
        aggregation=aggregation,
        min_num_voxel=min_num_voxel,
        min_threshold=min_threshold,
        mode=mode,
        save_seg=save_seg,
        save_iseg=save_iseg,
        stuff=stuff,
    )

    if num_workers > 0:
        with Pool(processes=num_workers) as p:
            p.starmap(_fn, zip(source, repeat(target_dir)))
    else:
        for s in maybe_verbose_iterable(source):
            _fn(s, target_dir)


def import_single_case(
    logits_source: Path,
    logits_target_dir: Optional[Path],
    aggregation: str,
    min_num_voxel: int,
    min_threshold: Optional[float],
    mode: str,
    save_seg: bool = True,
    save_iseg: bool = True,
    stuff: Optional[Sequence[int]] = None,
):
    """
    Process a single case

    Args:
        logits_source: path to nnunet prediction
        logits_target_dir: path to dir where result should be saved
        aggregation: aggregation method for probabilities.
        save_seg: save semantic segmentation
        save_iseg: save instance segmentation
        stuff: stuff classes to remove
    """
    assert logits_source.is_file(), f"Logits source needs to be a file, found {logits_source}"
    assert logits_target_dir.is_dir(), f"Logits target dir needs to be a dir, found {logits_target_dir}"

    case_name = logits_source.stem
    logger.info(f"Processing {case_name}")
    properties_file = logits_source.parent / f"{case_name}.pkl"
    probs = np.load(str(logits_source))["probabilities"]
    properties_dict = load_pickle(properties_file)

    res = softmax_to_instances(
        probs,
        aggregation=aggregation,
        min_num_voxel=min_num_voxel,
        min_threshold=min_threshold,
        mode=mode,
        stuff=stuff,
    )

    detection_target = logits_target_dir / f"{case_name}_boxes.pkl"
    segmentation_target = logits_target_dir / f"{case_name}_segmentation.pkl"
    instances_target = logits_target_dir / f"{case_name}_instances.pkl"

    boxes = {}
    for key in ["pred_boxes", "pred_labels", "pred_scores"]:
        if not isinstance(res[key], np.ndarray):
            boxes[key] = np.array(res[key])
        else:
            boxes[key] = res[key]
    boxes["original_size_of_raw_data"] = properties_dict["shape_before_cropping"]
    boxes["itk_origin"] = properties_dict["sitk_stuff"]["origin"]
    boxes["itk_direction"] = properties_dict["sitk_stuff"]["direction"]
    boxes["itk_spacing"] = properties_dict["sitk_stuff"]["spacing"]

    save_pickle(boxes, detection_target)
    if save_iseg:
        instances = {key: res[key] for key in ["pred_instances", "pred_labels", "pred_scores"]}
        save_pickle(instances, instances_target)
    if save_seg:
        segmentation = {"pred_seg": np.argmax(probs, axis=0)}
        save_pickle(segmentation, segmentation_target)


def nnunet_dataset_json(nnunet_task: str):
    if (p := os.getenv("nnUNet_raw_data_base")) is not None:
        search_dir = Path(p) / "nnUNet_raw_data" / nnunet_task
        logger.info(f"Looking for dataset.json in {search_dir}")
        if (fp := search_dir / "dataset.json").is_file():
            return load_json(fp)
    elif (p := os.getenv("nnUNet_preprocessed")) is not None:
        search_dir = Path(p) / nnunet_task
        logger.info(f"Looking for dataset.json in {search_dir}")
        if (fp := search_dir / "dataset.json").is_file():
            return load_json(fp)
    else:
        raise ValueError("Was not able to find nnunet dataset.json")


def copy_and_ensemble(cid, nnunet_dirs, nnunet_prediction_dir):
    logger.info(f"Copy and ensemble: {cid}")
    case = [
        np.load(_nnunet_dir / f"fold_{fold}" / "validation" / f"{cid}.npz")["probabilities"]
        for _nnunet_dir in nnunet_dirs
    ]
    assert len(case) == len(nnunet_dirs)
    case_ensemble = np.mean(case, axis=0)
    assert case_ensemble.shape == case[0].shape

    np.savez_compressed(nnunet_prediction_dir / f"{cid}.npz", probabilities=case_ensemble)


def copy_and_ensemble_test(cid, nnunet_dirs, nnunet_prediction_dir):
    logger.info(f"Copy and ensemble: {cid}")
    case = [np.load(_nnunet_dir / f"{cid}.npz")["probabilities"] for _nnunet_dir in nnunet_dirs]
    assert len(case) == len(nnunet_dirs)
    case_ensemble = np.mean(case, axis=0)
    assert case_ensemble.shape == case[0].shape

    np.savez_compressed(nnunet_prediction_dir / f"{cid}.npz", probabilities=case_ensemble)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--nnunet",
        type=Path,
        nargs="+",
        help="if val: Path to nnunet dir. e,g. "
        "../nnUNet/3d_fullres/TaskX/nnUNetTrainerV2__nnUNetPlansv2.1 "
        "if test: path to prediction dirs to ensemble. Val mode needed to be run before!",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        required=True,
        help="Provide operation mode. 'val' will ensemble and run "
        "empirical optimization. 'test' will load settings and postprocess.",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        help="detection task id, needed to determine stuff classes",
        required=True,
    )
    parser.add_argument(
        "-pf",
        "--prefix",
        type=str,
        default="val",
        help="Prefix for folder. One of 'val', 'test'",
        required=False,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
        help="Number of worker to use",
        required=False,
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Argmax with max probability aggregation.",
    )
    parser.add_argument(
        "--nnunet_model_name",
        type=str,
        default="nnUNet",
        help="Name of model which will be saved to nndetection directory",
        required=False,
    )

    # Evaluation related settings
    parser.add_argument("--save_seg", help="Save semantic segmentation", action="store_true")
    # parser.add_argument("--save_iseg", help="Save instance segmentation", action="store_true")

    args = parser.parse_args()
    nnunet_dirs = args.nnunet
    task = args.task
    prefix = args.prefix
    mode = args.mode
    num_workers = args.num_workers
    simple = args.simple
    nnunet_model_name = args.nnunet_model_name

    save_seg = args.save_seg
    save_iseg = False

    nnunet_dir = nnunet_dirs[0]
    task = get_task(task, name=True)

    task_dir = Path(os.getenv("det_models")) / task
    initialize_config_module(config_module="nndet.conf", version_base="1.1")
    cfg = compose(task, "config.yaml", overrides=[])

    if simple:
        nndet_unet_dir = task_dir / f"{nnunet_model_name}Basic" / "consolidated"
    else:
        nndet_unet_dir = task_dir / f"{nnunet_model_name}" / "consolidated"

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    log_file = nndet_unet_dir / "import.log"
    logger.add(log_file, level="INFO")

    instance_classes = cfg["data"]["labels"]
    stuff_classes = cfg.get("labels_stuff", {})
    num_instance_classes = len(instance_classes)
    stuff_classes = {str(int(key) + num_instance_classes): item for key, item in stuff_classes.items() if int(key) > 0}
    stuff = [int(s) for s in stuff_classes.keys()]

    if mode.lower() == "val":
        # validation
        nnunet_prediction_dir = nndet_unet_dir / "validation_raw_all"
        nnunet_prediction_dir.mkdir(parents=True, exist_ok=True)

        # copy all predictions from nnunet into one directory
        for fold in range(5):
            case_ids = [
                p.stem for p in (nnunet_dir / f"fold_{fold}" / "validation").iterdir() if p.name.endswith(".npz")
            ]
            logger.info(f"Copy and ensemble results fold {fold} with {len(case_ids)} cases.")

            # copy properties
            for p in [p for p in (nnunet_dir / f"fold_{fold}" / "validation").iterdir() if p.name.endswith(".pkl")]:
                shutil.copyfile(p, nnunet_prediction_dir / p.name)

            if num_workers > 0:
                with Pool(processes=max(num_workers // 4, 1)) as p:
                    p.starmap(
                        copy_and_ensemble,
                        zip(
                            case_ids,
                            repeat(nnunet_dirs),
                            repeat(nnunet_prediction_dir),
                        ),
                    )
            else:
                for cid in case_ids:
                    copy_and_ensemble(cid, nnunet_dirs, nnunet_prediction_dir)

        if simple:
            # Basic
            postprocessing_settings = {
                "aggregation": "max",
                "min_num_voxel": 5,
                "min_threshold": None,
            }
            save_pickle(postprocessing_settings, nndet_unet_dir / "postprocessing.pkl")
        else:
            # Plus
            postprocessing_settings = import_nnunet_boxes(
                nnunet_prediction_dir=nnunet_prediction_dir,
                save_dir=nndet_unet_dir,
                boxes_gt_dir=Path(os.getenv("det_data")) / task / "preprocessed" / "labelsTr",
                classes=list(cfg["data"]["labels"].keys()),
                stuff=stuff,
                num_workers=num_workers,
            )

        save_pickle({}, nndet_unet_dir / "plan.pkl")
        target_dir = nndet_unet_dir / "val_predictions"
    else:
        # inference
        case_ids = [p.stem for p in nnunet_dir.iterdir() if p.name.endswith(".npz")]
        nnunet_prediction_dir = nndet_unet_dir / "test_raw_all"
        nnunet_prediction_dir.mkdir(parents=True, exist_ok=True)

        if num_workers > 0:
            with Pool(processes=max(num_workers // 4, 1)) as p:
                p.starmap(
                    copy_and_ensemble_test,
                    zip(
                        case_ids,
                        repeat(nnunet_dirs),
                        repeat(nnunet_prediction_dir),
                    ),
                )
        else:
            for cid in case_ids:
                copy_and_ensemble_test(cid, nnunet_dirs, nnunet_prediction_dir)

        # copy properties
        for p in [p for p in nnunet_dir.iterdir() if p.name.endswith(".pkl")]:
            shutil.copyfile(p, nnunet_prediction_dir / p.name)

        postprocessing_settings = load_pickle(nndet_unet_dir / "postprocessing.pkl")
        target_dir = nndet_unet_dir / "test_predictions"

    logger.info("Creating final predictions")
    target_dir.mkdir(parents=True, exist_ok=True)
    import_dir(
        nnunet_prediction_dir=nnunet_prediction_dir,
        target_dir=target_dir,
        save_seg=save_seg,
        save_iseg=save_iseg,
        stuff=stuff,
        num_workers=num_workers,
        **postprocessing_settings,
    )
