# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import os
from pathlib import Path

import yaml
from hydra import compose as hydra_compose
from omegaconf import OmegaConf

from nndet.io.paths import get_task


def load_dataset_info(task_dir: os.PathLike) -> dict:
    """
    Load dataset information from a given task directory

    Args:
        task_dir: path to directory of specific task e.g. ../Task12_LIDC

    Returns:
        dict: loaded dataset info. Typically includes:
            `name` (str): name of dataset
            `target_class` (str)
    """
    task_dir = Path(task_dir)
    yaml_path = task_dir / "dataset.yaml"
    yaml_path_fallback = task_dir / "dataset.yml"
    json_path = task_dir / "dataset.json"

    if yaml_path.is_file():
        with open(yaml_path, "r") as f:
            data = yaml.full_load(f)
    elif yaml_path_fallback.is_file():
        with open(yaml_path_fallback, "r") as f:
            data = yaml.full_load(f)
    elif json_path.is_file():
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        raise RuntimeError(f"Did not find dataset.json or dataset.yaml in {task_dir}")
    return data


def compose(task, *args, models: bool = False, **kwargs) -> dict:
    cfg = hydra_compose(*args, **kwargs)
    OmegaConf.set_struct(cfg, False)
    task_name = get_task(task, name=True, models=models)
    cfg["task"] = task_name
    cfg["data"] = load_dataset_info(get_task(task_name))

    for imp in cfg.get("additional_imports", []):
        print(f"Additional import found {imp}")
        importlib.import_module(imp)
    return cfg
