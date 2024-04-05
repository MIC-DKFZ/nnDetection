# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import os
from pathlib import Path
from typing import List, Union

import yaml
from hydra import compose as hydra_compose
from loguru import logger
from omegaconf import OmegaConf

from nndet.io.load import load_json, load_pickle, save_json, save_pickle
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


def load_plan_from_task(plan_id: str, task: str) -> dict:
    """
    Load plan from preprocessed task directory

    Args:
        plan_id: plan identifier to load
        task: task identifier to load plan from
    """
    if plan_id.endswith(".pkl") or plan_id.endswith(".json") or plan_id.endswith(".pickle"):
        raise ValueError("Please provide the plan name without file extension")

    det_data = Path(os.getenv("det_data"))
    task_dir = det_data / task
    if not task_dir.is_dir():
        task = get_task(task, name=True)
        task_dir = det_data / task

    plan_json_path = task_dir / "preprocessed" / f"{plan_id}.json"
    plan_pkl_path = task_dir / "preprocessed" / f"{plan_id}.pkl"

    if plan_json_path.is_file():
        plan = load_json(plan_json_path)
    elif plan_pkl_path.is_file():
        logger.warning(f"Loading plan from {plan_pkl_path} which is deprected since nnDetV2")
        plan = load_pickle(plan_pkl_path)
    else:
        raise ValueError(f"Did not find plan {plan_id} in {task_dir}")
    return plan


def load_plan_from_model(
    task: str,
    model: str,
    fold: Union[str, int],
    plan_name: str = "plan",
) -> dict:
    """
    Load plan from model directory

    Args:
        task: task identifier to load plan from
        model: model identifier to load plan from
        fold: fold to load plan from
        plan_name: name of plan to load
    """
    if plan_name.endswith(".pkl") or plan_name.endswith(".json") or plan_name.endswith(".pickle"):
        raise ValueError("Please provide the plan name without file extension")

    det_models = Path(os.getenv("det_models"))
    task_dir = det_models / task
    if not task_dir.is_dir():
        task = get_task(task, name=True)
        task_dir = det_models / task

    if isinstance(fold, str):
        fold_str = fold
    else:
        fold_str = "consolidated" if fold == -1 else f"fold{fold}"
    model_dir = task_dir / model / fold_str

    plan_json_path = model_dir / f"{plan_name}.json"
    plan_pkl_path = model_dir / f"{plan_name}.pkl"

    if plan_json_path.is_file():
        plan = load_json(plan_json_path)
    elif plan_pkl_path.is_file():
        logger.warning(f"Loading plan from {plan_pkl_path} which is deprected since nnDetV2")
        plan = load_pickle(plan_pkl_path)
    else:
        raise ValueError(f"Did not find plan {plan_name} in {task_dir}")
    return plan


def load_plan_from_dir(plan_dir: os.PathLike, plan_name: str) -> dict:
    """
    Load plan from directory

    Args:
        plan_dir: directory to load plan from
        plan_name: name of plan to load

    Returns:
        dict: restored plan
    """
    if plan_name.endswith(".pkl") or plan_name.endswith(".json") or plan_name.endswith(".pickle"):
        raise ValueError("Please provide the plan name without file extension")

    plan_json_path = plan_dir / f"{plan_name}.json"
    plan_pkl_path = plan_dir / f"{plan_name}.pkl"

    if plan_json_path.is_file():
        plan = load_json(plan_json_path)
    elif plan_pkl_path.is_file():
        logger.warning(f"Loading plan from {plan_pkl_path} which is deprected since nnDetV2")
        plan = load_pickle(plan_pkl_path)
    else:
        raise ValueError(f"Did not find plan {plan_name} in {plan_dir}")
    return plan


def save_plan_to_model(
    plan: dict,
    task: str,
    model: str,
    fold: Union[str, int],
    save_name: str = "plan",
) -> None:
    """
    Save plan to model directory

    Args:
        plan: plan to save
        task: task identifier to save plan to
        model: model identifier to save plan to
        fold: fold to save plan to
        save_name: name of plan to save. Defaults to "plan".
    """
    if save_name.endswith(".pkl") or save_name.endswith(".json") or save_name.endswith(".pickle"):
        raise ValueError("Please provide the plan name without file extension")

    det_models = Path(os.getenv("det_models"))
    task_dir = det_models / task
    if not task_dir.is_dir():
        task = get_task(task, name=True)
        task_dir = det_models / task

    if isinstance(fold, str):
        fold_str = fold
    else:
        fold_str = "consolidated" if fold == -1 else f"fold{fold}"
    model_dir = task_dir / model / fold_str

    try:
        save_json(plan, model_dir / f"{save_name}")
    except TypeError:
        logger.warning(f"Saving plan to {model_dir / f'{save_name}'} which is deprected since nnDetV2")
        save_pickle(plan, model_dir / f"{save_name}")


def load_splits_from_task(splits_id: str, task: str) -> List[dict]:
    """
    Load splits from preprocessed task directory

    Args:
        splits_id: name of splits file
        task: task identifier to load splits from

    Returns:
        List[dict]: list with 'train' and 'val' keys containing the
            respective cases
    """
    if splits_id.endswith(".pkl") or splits_id.endswith(".json") or splits_id.endswith(".pickle"):
        raise ValueError("Please provide the splits name without file extension")

    det_data = Path(os.getenv("det_data"))
    task_dir = det_data / task
    if not task_dir.is_dir():
        task = get_task(task, name=True)
        task_dir = det_data / task

    splits_json_path = task_dir / "preprocessed" / f"{splits_id}.json"
    splits_pkl_path = task_dir / "preprocessed" / f"{splits_id}.pkl"

    if splits_json_path.is_file():
        splits = load_json(splits_json_path)
    elif splits_pkl_path.is_file():
        logger.warning(f"Loading splits from {splits_pkl_path} which is deprected since nnDetV2")
        splits = load_pickle(splits_pkl_path)
    else:
        raise ValueError(f"Did not find splits {splits_id} in {task_dir}")
    return splits


def load_splits_from_model(
    task: str,
    model: str,
    fold: Union[str, int],
    splits_name: str = "splits",
) -> List[dict]:
    """
    Load splits from model directory

    Args:
        task: task identifier to load splits from
        model: model identifier to load splits from
        fold: fold to load splits from
        splits_name: name of splits to load
    """
    if splits_name.endswith(".pkl") or splits_name.endswith(".json") or splits_name.endswith(".pickle"):
        raise ValueError("Please provide the splits name without file extension")

    det_models = Path(os.getenv("det_models"))
    task_dir = det_models / task
    if not task_dir.is_dir():
        task = get_task(task, name=True)
        task_dir = det_models / task

    if isinstance(fold, str):
        fold_str = fold
    else:
        fold_str = "consolidated" if fold == -1 else f"fold{fold}"
    model_dir = task_dir / model / fold_str

    splits_json_path = model_dir / f"{splits_name}.json"
    splits_pkl_path = model_dir / f"{splits_name}.pkl"

    if splits_json_path.is_file():
        splits = load_json(splits_json_path)
    elif splits_pkl_path.is_file():
        logger.warning(f"Loading splits from {splits_pkl_path} which is deprected since nnDetV2")
        splits = load_pickle(splits_pkl_path)
    else:
        raise ValueError(f"Did not find splits {splits_name} in {task_dir}")
    return splits


def load_splits_from_dir(splits_dir: os.PathLike, splits_name: str) -> dict:
    """
    Load splits from directory

    Args:
        splits_dir: directory to load splits from
        splits_name: name of splits to load

    Returns:
        dict: restored splits
    """
    if splits_name.endswith(".pkl") or splits_name.endswith(".json") or splits_name.endswith(".pickle"):
        raise ValueError("Please provide the splits name without file extension")

    splits_json_path = splits_dir / f"{splits_name}.json"
    splits_pkl_path = splits_dir / f"{splits_name}.pkl"

    if splits_json_path.is_file():
        splits = load_json(splits_json_path)
    elif splits_pkl_path.is_file():
        logger.warning(f"Loading splits from {splits_pkl_path} which is deprected since nnDetV2")
        splits = load_pickle(splits_pkl_path)
    else:
        raise ValueError(f"Did not find splits {splits_name} in {splits_dir}")
    return splits


def save_splits_to_model(
    splits: List[dict],
    task: str,
    model: str,
    fold: Union[str, int],
    save_name: str = "splits",
) -> None:
    """
    Save splits to model directory

    Args:
        splits: splits to save
        task: task identifier to save splits to
        model: model identifier to save splits to
        fold: fold to save splits to
        save_name: name of splits to save. Defaults to "splits".
    """
    if save_name.endswith(".pkl") or save_name.endswith(".json") or save_name.endswith(".pickle"):
        raise ValueError("Please provide the splits name without file extension")

    det_models = Path(os.getenv("det_models"))
    task_dir = det_models / task
    if not task_dir.is_dir():
        task = get_task(task, name=True)
        task_dir = det_models / task

    if isinstance(fold, str):
        fold_str = fold
    else:
        fold_str = "consolidated" if fold == -1 else f"fold{fold}"
    model_dir = task_dir / model / fold_str

    try:
        save_json(splits, model_dir / f"{save_name}")
    except TypeError:
        logger.warning(f"Saving splits to {model_dir / f'{save_name}'} which is deprected since nnDetV2")
        save_pickle(splits, model_dir / f"{save_name}")
