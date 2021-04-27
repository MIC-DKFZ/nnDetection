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
import os
import sys
from typing import Any, Mapping, Type, TypeVar

from omegaconf import OmegaConf
from loguru import logger
from pathlib import Path

from nndet.utils.info import env_guard
from nndet.planning import PLANNER_REGISTRY
from nndet.io import get_task, get_training_dir
from nndet.io.load import load_pickle
from nndet.inference.loading import load_all_models
from nndet.inference.helper import predict_dir


def run(cfg: dict,
        training_dir: Path,
        process: bool = True,
        num_models: int = None,
        num_tta_transforms: int = None,
        ):
    """
    Run inference pipeline

    Args:
        cfg: configurations
        training_dir: path to model directory
        process: preprocess test data
        num_models: number of models to use for ensemble; if None all Models
            are used
        num_tta_transforms: number of tta transformation; if None the maximum
            number of transformation is used
    """
    plan = load_pickle(training_dir / "plan_inference.pkl")

    preprocessed_output_dir = Path(cfg["host"]["preprocessed_output_dir"])
    prediction_dir = training_dir / "test_predictions"

    logger.remove()
    logger.add(sys.stdout, format="{level} {message}", level="INFO")
    logger.add(Path(training_dir) / "inference.log", level="INFO")

    if process:
        planner_cls = PLANNER_REGISTRY.get(plan["planner_id"])
        planner_cls.run_preprocessing_test(
            preprocessed_output_dir=preprocessed_output_dir,
            splitted_4d_output_dir=cfg["host"]["splitted_4d_output_dir"],
            plan=plan,
            num_processes=cfg["prep"]["num_processes_processing"],
        )

    prediction_dir.mkdir(parents=True, exist_ok=True)
    source_dir = preprocessed_output_dir / plan["data_identifier"] / "imagesTs"
    predict_dir(source_dir=source_dir,
                target_dir=prediction_dir,
                cfg=cfg,
                plan=plan,
                source_models=training_dir,
                num_models=num_models,
                num_tta_transforms=num_tta_transforms,
                model_fn=load_all_models,
                restore=True,
                # do_seg=True, # TODO: change this...
                )


def set_arg(cfg: Mapping, key: str, val: Any, force_args: bool) -> Mapping:
    """
    Check if value of config and given key match and handle approriately:
    If values match no action will be performend.
    If the values do not match and force_args is activated the value
    in the config will be overwritten.
    if the values do not match and force args is deactivatd a ValueError
    will be raised.

    Args:
        cfg: config to check and write values to
        key: key to check.
        val: Potentially new value.
        force_args: Enable if config value should be overwritten if values do
            not match.

    Returns:
        Type[dict]: config with potentially changed key
    """
    if key not in cfg:
        raise ValueError(f"{key} is not in config.")

    if cfg[key] != val:
        if force_args:
            logger.warning(f"Found different values for {key}, will overwrite {cfg[key]} with {val}")
            cfg[key] = val
        else:
            raise ValueError(f"Found different values for {key} and overwrite disabled."
                             f"Found {cfg[key]} but expected {val}.")
    return cfg


@env_guard
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument('model', type=str, help="model name, e.g. RetinaUNetV0")
    parser.add_argument('-f', '--fold', type=int, help="fold to use for prediction. -1 uses the consolidated model",
                        required=False, default=-1)
    parser.add_argument('-nmodels', '--num_models', type=int, default=None,
                        help="number of models for ensemble(per default all models will be used)."
                             "NOT usable by default -- will use all models inside the folder!",
                        required=False)
    parser.add_argument('-ntta', '--num_tta', type=int, default=None,
                        help="number of tta transforms (per default most tta are chosen)",
                        required=False)
    parser.add_argument('-o', '--overwrites', type=str, nargs='+',
                        help="overwrites for config file", default=None,
                        required=False)
    parser.add_argument('--no_preprocess', help="Preprocess test data", action='store_false')
    parser.add_argument('--force_args',
                        help=("When transferring models betweens tasks the name "
                        "and fold might differ from the original one. "
                        "This forces an overwrite to the passed in arguments of"
                        " this function. This can be dangerous!"), action='store_true')

    args = parser.parse_args()
    model = args.model
    fold = args.fold
    task = args.task
    num_models = args.num_models
    num_tta_transforms = args.num_tta
    ov = args.overwrites
    force_args = args.force_args

    task_name = get_task(task, name=True)
    task_model_dir = Path(os.getenv("det_models"))
    training_dir = get_training_dir(task_model_dir / task_name / model, fold)

    process = args.no_preprocess

    cfg = OmegaConf.load(str(training_dir / "config.yaml"))

    cfg = set_arg(cfg, "task", task_name, force_args=force_args)
    cfg["exp"] = set_arg(cfg["exp"], "fold", fold, force_args=force_args)
    cfg["exp"] = set_arg(cfg["exp"], "id", model, force_args=force_args)

    overwrites = ov if ov is not None else []
    overwrites.append("host.parent_data=${env:det_data}")
    overwrites.append("host.parent_results=${env:det_models}")
    cfg.merge_with_dotlist(overwrites)

    run(OmegaConf.to_container(cfg, resolve=True),
        training_dir,
        process=process,
        num_models=num_models,
        num_tta_transforms=num_tta_transforms,
        )


if __name__ == '__main__':
    main()
