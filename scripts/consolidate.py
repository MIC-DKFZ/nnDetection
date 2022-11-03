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

import importlib
import argparse
import shutil
import sys
import os
from pathlib import Path
from typing import Sequence

from loguru import logger
from nndet.utils.check import env_guard
from omegaconf import OmegaConf

from nndet.ptmodule import MODULE_REGISTRY
from nndet.inference.sweeper import BoxSweeper
from nndet.inference.loading import get_latest_model
from nndet.inference.ensembler.base import extract_results
from nndet.io import get_task, load_pickle, save_pickle


def consolidate_models(source_dirs: Sequence[Path], target_dir: Path, ckpt: str):
    """
    Copy final models from folds into consolidated folder

    Args:
        source_dirs: directory of each fold to consolidate
        target_dir: directory to save models to
        ckpt: checkpoint identifier to select models for ensembling
    """
    for fold, sd in enumerate(source_dirs):
        model_paths = list(sd.glob('*.ckpt'))
        found_models = [mp for mp in model_paths if ckpt in str(mp.stem)]
        assert len(found_models) == 1, f"Found wrong number of models, {found_models}"
        model_path = found_models[0]
        assert f"fold{fold}" in str(model_path.parent.stem), f"Expected fold {fold} but found {model_path}"
        shutil.copy2(model_path, target_dir / f"model_fold{fold}.ckpt")


def consolidate_predictions(
    source_dirs: Sequence[Path],
    target_dir: Path,
    consolidate: str,
    ):
    """
    Consolidate sweep states to find new postprocessing hyperparameters

    Args:
        source_dirs: directory of each fold
        target_dir: directory of condolidated models
        consolidate: consolidation mode
    """
    if consolidate == 'export':
        logger.info("Consolidating sweep states for refinement.")
        postfix = "sweep_predictions"
    elif consolidate == 'copy':
        logger.info("Consolidating val predictions for evaluation")
        postfix = "val_predictions"
    else:
        raise ValueError(f"Consolidation {consolidate} is not supported")
    pred_dir = target_dir / postfix
    pred_dir.mkdir(parents=True, exist_ok=True)
    for source_dir in source_dirs:
        for p in [p for p in (source_dir / postfix).iterdir() if p.is_file()]:
            shutil.copy(p, pred_dir)


@env_guard
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str,
                        help="Task id e.g. Task12_LIDC OR 12 OR LIDC",
                        )
    parser.add_argument('model', type=str,
                        help="model name, e.g. RetinaUNetV0",
                        )
    parser.add_argument('-o', '--overwrites', type=str, nargs='+', required=False,
                        help="overwrites for config file. Only needed in case of box eval",
                        )
    parser.add_argument('-c', '--consolidate', type=str, default="export", required=False,
                        help=("Determines how to consolidate predictions: 'export' or 'copy'. "
                              "'copy' will copy the predictions of each fold into the directory for evaluation. "
                              "'export' will use the updated parameters after consolidation to update the "
                              "predictions and export them. This is only supported if one of the "
                              "sweep settings is active! Default: export"),
                        )
    parser.add_argument('--num_folds', type=int, default=5, required=False,
                        help="Number of folds. Default: 5",
                        )
    parser.add_argument('--no_model', action="store_false",
                        help="Deactivate if consolidating nnUNet results",
                        )
    parser.add_argument('--sweep_boxes', action="store_true",
                        help="Sweep for best parameters for bounding box based models",
                        )
    parser.add_argument('--sweep_instances', action="store_true",
                        help="Sweep for best parameters for instance segmentation based models",
                        )
    parser.add_argument('--ckpt', type=str, default="last", required=False,
                        help="Define identifier of checkpoint for consolidation. "
                        "Use this with care!")

    args = parser.parse_args()
    model = args.model
    task = args.task
    ov = args.overwrites

    consolidate = args.consolidate
    num_folds = args.num_folds
    do_model_consolidation = args.no_model

    sweep_boxes = args.sweep_boxes
    sweep_instances = args.sweep_instances
    ckpt = args.ckpt

    if consolidate == "export" and not (sweep_boxes or sweep_instances):
        raise ValueError("Export needs new parameter sweep! Actiate one of the sweep "
                         "arguments or change to copy mode")

    task_dir = Path(os.getenv("det_models")) / get_task(task, name=True, models=True)
    model_dir = task_dir / model
    if not model_dir.is_dir():
        raise ValueError(f"{model_dir} does not exist")
    target_dir = model_dir / "consolidated"

    logger.remove()
    logger.add(
        sys.stdout,
        format="<level>{level} {message}</level>",
        level="INFO",
        colorize=True,
        )
    logger.add(Path(target_dir) / "consolidate.log", level="DEBUG")

    logger.info(f"looking for models in {model_dir}")
    training_dirs = [get_latest_model(model_dir, fold) for fold in range(num_folds)]
    logger.info(f"Found training dirs: {training_dirs}")

    # model consolidation
    if do_model_consolidation:
        logger.info("Consolidate models")
        if ckpt != "last":
            logger.warning(f"Found ckpt overwrite {ckpt}, this is not the default, "
                           "this can drastically influence the performance!")
        consolidate_models(training_dirs, target_dir, ckpt)

    # consolidate predictions
    logger.info("Consolidate predictions")
    consolidate_predictions(
        source_dirs=training_dirs,
        target_dir=target_dir,
        consolidate=consolidate,
        )

    shutil.copy2(training_dirs[0] / "plan.pkl", target_dir)
    shutil.copy2(training_dirs[0] / "config.yaml", target_dir)

    # invoke new parameter sweeps
    cfg = OmegaConf.load(str(target_dir / "config.yaml"))
    ov = ov if ov is not None else []
    ov.append("host.parent_data=${oc.env:det_data}")
    ov.append("host.parent_results=${oc.env:det_models}")
    if ov is not None:
        cfg.merge_with_dotlist(ov)

    for imp in cfg.get("additional_imports", []):
        print(f"Additional import found {imp}")
        importlib.import_module(imp)

    preprocessed_output_dir = Path(cfg["host"]["preprocessed_output_dir"])
    plan = load_pickle(target_dir / "plan.pkl")
    gt_dir = preprocessed_output_dir / plan["data_identifier"] / "labelsTr"

    if sweep_boxes:
        logger.info("Sweeping box predictions")
        module = MODULE_REGISTRY[cfg["module"]]
        ensembler_cls = module.get_ensembler_cls(
            key="boxes", dim=plan["network_dim"])  # TODO: make this configurable

        sweeper = BoxSweeper(
            classes=[item for _, item in cfg["data"]["labels"].items()],
            pred_dir=target_dir / "sweep_predictions",
            gt_dir=gt_dir,
            target_metric=cfg["trainer_cfg"].get("eval_score_key",
                                                 "mAP_IoU_0.10_0.50_0.05_MaxDet_100"),
            ensembler_cls=ensembler_cls,
            save_dir=target_dir / "sweep",
        )
        inference_plan = sweeper.run_postprocessing_sweep()
    elif sweep_instances:
        raise NotImplementedError

    plan = load_pickle(target_dir / "plan.pkl")
    if consolidate != 'copy':
        plan["inference_plan"] = inference_plan
        save_pickle(plan, target_dir / "plan_inference.pkl")

        for restore in [True, False]:
            export_dir = target_dir / "val_predictions" if restore else \
                target_dir / "val_predictions_preprocessed"
            extract_results(
                source_dir=target_dir / "sweep_predictions",
                target_dir=export_dir,
                ensembler_cls=ensembler_cls,
                restore=restore,
                **inference_plan,
                )
    else:
        logger.warning("Plan used from fold 0, not updated with consolidation")
        save_pickle(plan, target_dir / "plan_inference.pkl")

if __name__ == '__main__':
    main()
