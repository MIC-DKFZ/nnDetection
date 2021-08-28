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

import argparse
from nndet.io.load import save_json
import os
import sys
import shutil
from functools import partial
from itertools import repeat
from multiprocessing import Pool

from pathlib import Path, PurePath
from typing import Union, Sequence, Optional

import numpy as np

from hydra import initialize_config_module
from loguru import logger

from nndet.evaluator.registry import evaluate_box_dir
from nndet.io import load_pickle, save_pickle, get_task, load_json
from nndet.utils.clustering import softmax_to_instances
from nndet.utils.config import compose
from nndet.utils.info import maybe_verbose_iterable

Pathlike = Union[str, Path]
TARGET_METRIC = "mAP_IoU_0.10_0.50_0.05_MaxDet_100"


def import_nnunet_boxes(
        # settings
        nnunet_prediction_dir: Pathlike,
        save_dir: Pathlike,
        boxes_gt_dir: Pathlike,
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
    nnunet_prediction_dir: Pathlike,
    target_dir: Optional[Pathlike] = None,
    aggregation="max",
    min_num_voxel=0,
    min_threshold=None,
    save_seg: bool = True,
    save_iseg: bool = True,
    stuff: Optional[Sequence[int]] = None,
    num_workers: int = 6,
):
    source = [f for f in nnunet_prediction_dir.iterdir() if f.suffix == ".npz"]

    _fn = partial(import_single_case,
                  aggregation=aggregation,
                  min_num_voxel=min_num_voxel,
                  min_threshold=min_threshold,
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


def import_single_case(logits_source: Path,
                       logits_target_dir: Optional[Path],
                       aggregation: str,
                       min_num_voxel: int,
                       min_threshold: Optional[float],
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
    probs = np.load(str(logits_source))["softmax"]

    properties_dict = load_pickle(properties_file)
    bbox = properties_dict.get('crop_bbox')
    shape_original_before_cropping = properties_dict.get('original_size_of_raw_data')

    if bbox is not None:
        tmp = np.zeros((probs.shape[0], *shape_original_before_cropping))
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + probs.shape[c + 1], shape_original_before_cropping[c]))

        tmp[:, bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = probs
        probs = tmp

    res = softmax_to_instances(probs,
                               aggregation=aggregation,
                               min_num_voxel=min_num_voxel,
                               min_threshold=min_threshold,
                               stuff=stuff,
                               )

    detection_target = logits_target_dir / f"{case_name}_boxes.pkl"
    segmentation_target = logits_target_dir / f"{case_name}_segmentation.pkl"
    instances_target = logits_target_dir / f"{case_name}_instances.pkl"

    boxes = {key: res[key] for key in ["pred_boxes", "pred_labels", "pred_scores"]}
    boxes["original_size_of_raw_data"] = properties_dict["original_size_of_raw_data"]
    boxes["itk_origin"] = properties_dict["itk_origin"]
    boxes["itk_direction"] = properties_dict["itk_direction"]
    boxes["itk_spacing"] = properties_dict["itk_spacing"]

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
    case = [np.load(_nnunet_dir / f"fold_{fold}" / "validation_raw" / f"{cid}.npz")["softmax"] for _nnunet_dir in nnunet_dirs]
    assert len(case) == len(nnunet_dirs)
    case_ensemble = np.mean(case, axis=0)
    assert case_ensemble.shape == case[0].shape
    
    np.savez_compressed(nnunet_prediction_dir / f"{cid}.npz", softmax=case_ensemble)


def copy_and_ensemble_test(cid, nnunet_dirs, nnunet_prediction_dir):
    logger.info(f"Copy and ensemble: {cid}")
    case = [np.load(_nnunet_dir / f"{cid}.npz")["softmax"] for _nnunet_dir in nnunet_dirs]
    assert len(case) == len(nnunet_dirs)
    case_ensemble = np.mean(case, axis=0)
    assert case_ensemble.shape == case[0].shape
    
    np.savez_compressed(nnunet_prediction_dir / f"{cid}.npz", softmax=case_ensemble)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--nnunet', type=Path, nargs='+',
                        help='if val: Path to nnunet dir. e,g. '
                             '../nnUNet/3d_fullres/TaskX/nnUNetTrainerV2__nnUNetPlansv2.1 '
                             'if test: path to prediction dirs to ensemble. Val mode needed to be run before!',
                        required=True,
                        )
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help="Provide operation mode. 'val' will ensemble and run "
                        "empirical optimization. 'test' will load settings and postprocess.")
    parser.add_argument('-t', '--task', type=str, default=None,
                        help="detection task id, needed to determine stuff classes"
                             "If it is not provided via an argument the script tries to determine "
                             "it from the nnunet path, this works only if the task names are identical!"
                              "Need to provide task id in test mode!",
                        required=False,
                        )
    parser.add_argument('-pf', '--prefix', type=str, default='val',
                        help="Prefix for folder. One of 'val', 'test'",
                        required=False,
                        )
    parser.add_argument('--num_workers', type=int, default=6,
                        help="Number of worker to use",
                        required=False,
                        )
    parser.add_argument('--simple', action='store_true',
                        help="Argmax with max probability aggregation.",
                        )
    # Evaluation related settings
    parser.add_argument('--save_seg', help="Save semantic segmentation", action='store_true')
    parser.add_argument('--save_iseg', help="Save instance segmentation", action='store_true')

    args = parser.parse_args()
    nnunet_dirs = args.nnunet
    task = args.task
    prefix = args.prefix
    mode = args.mode
    num_workers = args.num_workers
    simple = args.simple

    save_seg = args.save_seg
    save_iseg = args.save_iseg

    nnunet_dir = nnunet_dirs[0]
    if task is None:
        # select corresponding nnDetection task
        task_names = [n for n in PurePath(nnunet_dir).parts if "Task" in n]
        if len(task_names) > 1:
            logger.error(f"Found multiple task names trying to continue with {task_names[-1]}")
        if len(task_names) == 0:
            logger.error(f"Could not derive task name from path please use "
                         "-t/--task to provide the name via cmd line!")
        logger.info(f"Found nnunet task {task_names[-1]} in nnunet path")
        nnunet_task = task_names[-1]

        logger.info(f"Using nnunet task {nnunet_task} as detection task id")
        task = nnunet_task
    else:
        task = get_task(task, name=True)

    task_dir = Path(os.getenv("det_models")) / task
    initialize_config_module(config_module="nndet.conf")
    cfg = compose(task, "config.yaml", overrides=[])

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    log_file = task_dir / "nnUNet" / "import.log"
    logger.add(log_file, level="INFO")
    
    if simple:
        nndet_unet_dir = task_dir / "nnUNet_Simple" / "consolidated"
    else:
        nndet_unet_dir = task_dir / "nnUNet" / "consolidated"

    instance_classes = cfg["data"]["labels"]
    stuff_classes = cfg.get("labels_stuff", {})
    num_instance_classes = len(instance_classes)
    stuff_classes = {
                str(int(key) + num_instance_classes): item
                for key, item in stuff_classes.items() if int(key) > 0
            }
    stuff = [int(s) for s in stuff_classes.keys()]

    if mode.lower() == "val":
        nnunet_prediction_dir = nndet_unet_dir /f"validation_raw_all"
        nnunet_prediction_dir.mkdir(parents=True, exist_ok=True)

        # copy all predictions from nnunet into one directory
        for fold in range(5):
            case_ids = [p.stem for p in (nnunet_dir / f"fold_{fold}" / "validation_raw").iterdir() if p.name.endswith(".npz")]
            logger.info(f"Copy and ensemble results fold {fold} with {len(case_ids)} cases.")

            # copy properties
            for p in [p for p in (nnunet_dir / f"fold_{fold}" / "validation_raw").iterdir() if p.name.endswith(".pkl")]:
                shutil.copyfile(p, nnunet_prediction_dir / p.name)

            if num_workers > 0:
                with Pool(processes=max(num_workers // 4, 1)) as p:
                    p.starmap(copy_and_ensemble,
                            zip(case_ids,
                                repeat(nnunet_dirs),
                                repeat(nnunet_prediction_dir),
                                ))
            else:
                for cid in case_ids:
                    copy_and_ensemble(cid, nnunet_dirs, nnunet_prediction_dir)

        if simple:
            postprocessing_settings = {
                "aggregation": "max",
                "min_num_voxel": 5,
                "min_threshold": None,
            }
            save_pickle(postprocessing_settings, nndet_unet_dir / "postprocessing.pkl")
        else:
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
        case_ids = [p.stem for p in nnunet_dir.iterdir() if p.name.endswith(".npz")]
        nnunet_prediction_dir = nndet_unet_dir /f"test_raw_all"
        nnunet_prediction_dir.mkdir(parents=True, exist_ok=True)
        
        if num_workers > 0:
                with Pool(processes=max(num_workers // 4, 1)) as p:
                    p.starmap(copy_and_ensemble_test,
                              zip(case_ids,
                                  repeat(nnunet_dirs),
                                  repeat(nnunet_prediction_dir),
                                  ))
        else:
            for cid in case_ids:
                copy_and_ensemble_test(cid, nnunet_dirs, nnunet_prediction_dir)

        # copy properties
        for p in [p for p in nnunet_dir.iterdir() if p.name.endswith(".pkl")]:
            shutil.copyfile(p, nnunet_prediction_dir / p.name)

        postprocessing_settings = load_pickle(nndet_unet_dir / "postprocessing.pkl")
        target_dir = nndet_unet_dir / "test_predictions"

    logger.info(f"Creating final predictions")
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
