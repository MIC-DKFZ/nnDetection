# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from loguru import logger

from nndet.io.load import load_pickle
from nndet.utils.info import maybe_verbose_iterable


def predict_dir(
    source_dir: os.PathLike,
    target_dir: os.PathLike,
    cfg: dict,
    plan: dict,
    source_models: Path,
    model_fn: Callable[[Path, dict, dict, int], Sequence[dict]],
    num_models: int = None,
    num_tta_transforms: int = None,
    restore: bool = False,
    case_ids: Optional[Sequence[str]] = None,
    save_state: bool = False,
    **kwargs,
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

    models = model_fn(
        source_models=source_models,
        cfg=cfg,
        plan=plan,
        num_models=num_models,
    )
    predictor = models[0]["model"].get_predictor(
        plan=plan,
        models=[m["model"] for m in models],
        num_tta_transforms=num_tta_transforms,
        **kwargs,
    )

    if case_ids is None:
        case_paths = list(source_dir.glob("*.npz"))
        case_paths = [cp for cp in case_paths if "_gt.npz" not in str(cp)]
    else:
        case_paths = [source_dir / f"{cid}.npz" for cid in case_ids]
    logger.info(f"Found {len(case_paths)} files for inference.")

    for idx, path in enumerate(case_paths, start=1):
        logger.info(f"Predicting case {idx} of {len(case_paths)}.")
        case_id = path.stem
        if path.is_file():
            case = np.load(str(path), allow_pickle=True)["data"]
        else:
            case = np.load(str(path)[:-4] + ".npy", allow_pickle=True)
        properties = load_pickle(path.parent / f"{case_id}.pkl")
        properties["transpose_backward"] = plan["transpose_backward"]

        if save_state:
            _ = predictor.predict_case(
                {"data": case},
                properties,
                save_dir=target_dir,
                case_id=case_id,
                restore=restore,
            )
        else:
            result = predictor.predict_case(
                {"data": case},
                properties,
                save_dir=None,
                case_id=None,
                restore=restore,
            )
            predictor.save_case(
                result=result,
                target_dir=target_dir,
                case_id=case_id,
            )
    return predictor


def extract_results(
    source_dir: os.PathLike,
    target_dir: os.PathLike,
    ensembler_cls: Callable,
    restore: bool,
    **params,
) -> None:
    """
    Compute case result from ensembler and save it

    Args:
        source_dir: directory which contains the saved predictions/state from
            the ensembler class
        target_dir: directory to save results
        ensembler_cls: ensembler class for prediction
        restore: if true, the results are converted into the opriginal image
            space
    """
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    for case_id in maybe_verbose_iterable(ensembler_cls.get_case_ids(source_dir)):
        ensembler = ensembler_cls.from_checkpoint(base_dir=source_dir, case_id=case_id)
        ensembler.update_parameters(**params)

        pred = ensembler.get_case_result(restore=restore)
        ensembler.save_result(
            data=pred,
            target_dir=Path(target_dir),
            case_name=case_id,
        )
