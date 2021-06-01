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

from pathlib import Path
from typing import Sequence, List, Dict, Callable, Optional

import numpy as np
from loguru import logger

from nndet.utils.tensor import to_numpy
from nndet.io.load import load_pickle, save_pickle
from nndet.io.paths import Pathlike, get_case_id_from_path
from nndet.inference.loading import load_final_model


def predict_dir(
    source_dir: Pathlike,
    target_dir: Pathlike,
    cfg: dict,
    plan: dict,
    source_models: Path,
    model_fn: Callable[[Path, dict, dict, int], Sequence[dict]] = load_final_model,
    num_models: int = None,
    num_tta_transforms: int = None,
    restore: bool = False,
    case_ids: Optional[Sequence[str]] = None,
    save_state: bool = False,
    **kwargs
    ):
    """
    Predict all preprocessed(!) cases inside a directory

    Args:
        source_dir: directory where preprocessed cases are located
        target_dir: directory to save predictions to
        cfg: config
            `predictor`: define predictor to use
        plan: plan
        source_models: directory where models for prediction are located
        model_fn: function to load model from directory
        num_models: number of models to use for prediction; None = all
        num_tta_transforms: number of tta transforms to use for
            prediction; None = all
        stage: current stage to predict
        restore: restore predictions in original image space
        case_ids: case ids to predict. If None the whole folder will be
            predicted
        save_state: If `true` the state of the ensembler is saved. If
            `false` only the final result is saved.
        kwargs: passed to :method:'get_predictor' method of module
    """
    logger.info("Running inference")

    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    models = model_fn(source_models, cfg, plan, num_models)
    predictor = models[0]["model"].get_predictor(
        plan=plan,
        models=[m["model"] for m in models],
        num_tta_transforms=num_tta_transforms,
        **kwargs,
    )

    if case_ids is None:
        case_paths = list(source_dir.glob('*.npz'))
        case_paths = [cp for cp in case_paths if "_gt.npz" not in str(cp)]
    else:
        case_paths = [source_dir / f"{cid}.npz" for cid in case_ids]
    logger.info(f"Found {len(case_paths)} files for inference.")

    for idx, path in enumerate(case_paths, start=1):
        logger.info(f"Predicting case {idx} of {len(case_paths)}.")
        case_id = get_case_id_from_path(str(path), remove_modality=False)
        if path.is_file():
            case = np.load(str(path), allow_pickle=True)['data']
        else:
            case = np.load(str(path)[:-4] + ".npy", allow_pickle=True)
        properties = load_pickle(path.parent / f"{case_id}.pkl")
        properties["transpose_backward"] = plan["transpose_backward"]

        if save_state:
            _ = predictor.predict_case({"data": case},
                                       properties,
                                       save_dir=target_dir,
                                       case_id=case_id,
                                       restore=restore,
                                       )
        else:
            result = predictor.predict_case({"data": case},
                                            properties,
                                            save_dir=None,
                                            case_id=None,
                                            restore=restore,
                                            )
            for key, item in to_numpy(result).items():
                save_pickle(item, target_dir / f"{case_id}_{key}.pkl")
    return predictor
