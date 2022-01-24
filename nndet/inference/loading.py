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

import os
from functools import partial
from pathlib import Path
from typing import Sequence, Optional

import torch
from loguru import logger

from nndet.io.paths import Pathlike


def get_loader_fn(mode: str, **kwargs):
    if mode.lower() == "all":
        load_fn = load_all_models
    else:
        load_fn = partial(load_final_model, identifier=mode, **kwargs)
    return load_fn


def get_latest_model(base_dir: Pathlike, fold: int = 0) -> Optional[Path]:
    """
    Get the latest training dir in a given base dir
    E.g. ../RetinaUNetV0/fold0__0, ../RetinaUNetV0/fold0__1
    -> would select fold0__1

    Args:
        base_dir: path to base dir
        fold: fold to look for

    Returns:
        Optional[Path]: If no model for specified fold is found, None this
            will return None
    """
    base_dir = Path(base_dir)
    m = [m for m in base_dir.iterdir() if m.is_dir()]
    m = [_m for _m in m if f"fold{fold}" in _m.stem]
    if m:
        return sorted(m, key=lambda x: x.stem, reverse=True)[0]
    else:
        return None


def load_final_model(
    source_models: Path,
    cfg: dict,
    plan: dict,
    num_models: int = 1,
    identifier: str = "last",
    ) -> Sequence[dict]:
    """
    Load final model from training

    Args:
        source_models: path to directory where models are saved
        cfg: config used for experiment
            `model`: name of model in DETECTION_REGISTRY
        plan: plan used for training
        num_models: Only supports one model
        identifier: looks for identifier inside of model name

    Returns:
        Sequence[dict]: loaded models
            `model`: loaded model
            `rank`: rank is always 0
    """
    from nndet.ptmodule import MODULE_REGISTRY

    assert num_models == 1, f"load_final_model only supports num_models=1, found {num_models}"
    logger.info(f"Loading {identifier} model")

    model_names = list(source_models.glob('*.ckpt'))
    model_names = [m for m in model_names if identifier in str(m.stem)]
    assert len(model_names) == 1, f"Found wrong number of models, {model_names} in {source_models} with {identifier}"

    path = model_names[0]
    model = MODULE_REGISTRY[cfg["module"]](
        model_cfg=cfg["model_cfg"],
        trainer_cfg=cfg["trainer_cfg"],
        plan=plan,
        )
    state_dict = torch.load(path, map_location="cpu")["state_dict"]
    t = model.load_state_dict(state_dict)
    logger.info(f"Loaded {path} with {t}")
    model.float()
    model.eval()
    return [{"model": model, "rank": 0}]


def load_all_models(
    source_models: Path, 
    cfg: dict, 
    plan: dict,
    *args, 
    **kwargs,
    ):
    """
    Load all models to ensemble

    Args:
        source_models: path to directory where models are saved
        cfg: config used for experiment
            `model`: name of model in DETECTION_REGISTRY
        plan: plan used for training
        kwargs: not used

    Returns:
        Sequence[dict]: loaded models
            `model`: loaded model
            `rank`: rank of model
    """
    from nndet.ptmodule import MODULE_REGISTRY

    model_names = list(source_models.glob('*.ckpt'))
    if not model_names:
        raise RuntimeError(f"Did not find any models in {source_models}")
    logger.info(f"Found {len(model_names)} models to ensemble")

    models = []
    for path in model_names:
        model = MODULE_REGISTRY[cfg["module"]](
            model_cfg=cfg["model_cfg"],
            trainer_cfg=cfg["trainer_cfg"],
            plan=plan,
            )
        state_dict = torch.load(path, map_location="cpu")["state_dict"]
        t = model.load_state_dict(state_dict)
        logger.info(f"Loaded {path} with {t}")
        model.float()
        model.eval()
        models.append({"model": model.cpu()})
    return models
