# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import argparse
import datetime
import importlib
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from loguru import logger
from omegaconf import OmegaConf

from nndet.inference.helper import predict_dir
from nndet.inference.loading import load_all_models, load_best_model, load_last_model
from nndet.io import get_task, get_training_dir
from nndet.io.load import load_pickle
from nndet.planning import PLANNER_REGISTRY
from nndet.utils.check import env_guard
from nndet.utils.enums import LoadModels


def _setup_logger(log_dir: os.PathLike):
    """
    Create logger

    Args:
        log_dir: directory to write logfile to
    """
    log_dir = Path(log_dir)

    logger.remove()
    logger.add(
        sys.stdout,
        format="<level>{level} {message}</level>",
        level="INFO",
        colorize=True,
    )
    logger.add(log_dir / "inference.log", level="INFO")

    current_time = datetime.now()
    current_time_str = current_time.strftime("%d/%m/%Y %H:%M:%S")
    logger.info(f"+++ Running nndet_inference {current_time_str} +++")


def _preprocess(
    training_dir: os.PathLike,
    raw_splitted_dir: os.PathLike,
    preprocessed_dir: os.PathLike,
    num_processes: int,
) -> str:
    """
    Function to run preprocessing on test data

    Args:
        training_dir: cirectory containing the model which should be predicted
            afterwards. Specifically, the plan file is used here.
        raw_splitted_dir: directory containing the data to predict in nndet
            format (i.e. with _0000.nii.gz) ending. Data needs to have the same
            format as training data!
        preprocessed_dir: directory where preprocessed data is placed.
            Specifically, data is saved in
            `preprocessed_dir/data_identifier/imagesTs`
        num_processes: number of processes to use for preprocessing

    Returns:
        str: data identifier
    """
    training_dir = Path(training_dir)
    raw_splitted_dir = Path(raw_splitted_dir)
    preprocessed_dir = Path(preprocessed_dir)

    if not training_dir.is_dir():
        raise ValueError(f"Training dir {training_dir} needs exist.")
    if not raw_splitted_dir.is_dir():
        raise ValueError(f"Raw splitted dir {raw_splitted_dir} need to exist.")
    if preprocessed_dir.is_dir():
        print(
            f"Warning: Preprocessed dir {preprocessed_dir} already exists, "
            "files inside might get overwritten during preprocessing."
        )
    preprocessed_dir.mkdir(exist_ok=True, parents=True)

    _setup_logger(raw_splitted_dir)

    # load plan
    plan_inference_path = training_dir / "plan_inference.pkl"
    if not plan_inference_path.is_file():
        raise RuntimeError(
            f"Expected {plan_inference_path} to contain the plan for "
            "running inference. Either run nndet_consolidate to predict "
            "ensembles or nndet_sweep for single fold models."
        )
    plan = load_pickle(plan_inference_path)

    # load config
    config_path = training_dir / "config.yaml"
    if not config_path.is_file():
        raise RuntimeError(f"Expected {config_path} to contain the config for running inference.")
    cfg = OmegaConf.load(training_dir / "config.yaml")

    for imp in cfg.get("additional_imports", []):
        logger.info(f"Additional import found {imp}")
        importlib.import_module(imp)

    # run preprocessing
    logger.info(f"++ Running preprocessing of {raw_splitted_dir} saving" f" into {preprocessed_dir} ++")
    planner_cls = PLANNER_REGISTRY.get(plan["planner_id"])

    planner_cls.run_preprocessing_test(
        splitted_4d_output_dir=raw_splitted_dir,
        preprocessed_output_dir=preprocessed_dir,
        plan=plan,
        num_processes=num_processes,
    )
    data_identifier = plan["data_identifier"]
    logger.info(
        "++ Finished preprocessing, results located in " f"{preprocessed_dir / data_identifier / 'imagesTs'} ++"
    )
    return data_identifier


def _predict(
    training_dir: os.PathLike,
    preprocessed_images_dir: os.PathLike,
    prediction_dir: os.PathLike,
    num_tta_transforms: int,
    overwrites: Sequence[Any],
    load_models: LoadModels,
    batch_size: Optional[int] = None,
    case_ids: Optional[Sequence[str]] = None,
) -> None:
    """
    Run prediction of preprocessed data

    Args:
        training_dir: training directory containing plan_inference, config and
            model weights
        preprocessed_images_dir: directory containing preprocessed data
        prediction_dir: directory to save prediction into
        num_tta_transforms: number of TTA transforms to perform during
            inference
        overwrites: overwrites applied to config. Does not include
            overwrites to plan! (plan includes infos like batch size
            and path size)
        load_models: Define model weights, one of all | last | best
        batch_size: Optionally overwrite batch size during inference.
            Defaults to None.
        case_ids: Optionally provide case ids which should be predicted.
            Defaults to None.
    """
    training_dir = Path(training_dir)
    preprocessed_images_dir = Path(preprocessed_images_dir)
    prediction_dir = Path(prediction_dir)

    if not training_dir.is_dir():
        raise ValueError(f"Training dir {training_dir} needs to exist.")
    if not preprocessed_images_dir.is_dir():
        raise ValueError(f"Preprocessed images dir {preprocessed_images_dir} needs to exist.")
    prediction_dir.mkdir(parents=True, exist_ok=True)
    _setup_logger(prediction_dir)

    # load config
    config_path = training_dir / "config.yaml"
    if not config_path.is_file():
        raise RuntimeError(f"Expected {config_path} to contain the config for running inference.")
    cfg = OmegaConf.load(training_dir / "config.yaml")
    cfg.merge_with_dotlist(overwrites)

    for imp in cfg.get("additional_imports", []):
        logger.info(f"Additional import found {imp}")
        importlib.import_module(imp)

    # pop some unnecessary information as a safety measure
    cfg.pop("task")
    cfg.pop("exp")
    cfg.pop("host")

    # load plan
    plan_inference_path = training_dir / "plan_inference.pkl"
    if not plan_inference_path.is_file():
        raise RuntimeError(
            f"Expected {plan_inference_path} to contain the plan for "
            "running inference. Either run nndet_consolidate to predict "
            "ensembles or nndet_sweep for single fold models."
        )
    plan = load_pickle(plan_inference_path)

    if batch_size is not None:
        logger.info(
            f"Found batch size {batch_size} provided by inference script, "
            f"running inference with provided batch size."
        )
        plan["batch_size"] = batch_size

    # select model
    if load_models == LoadModels.ALL:
        load_models_fn = load_all_models
    elif load_models == LoadModels.LAST:
        load_models_fn = load_last_model
    elif load_models == LoadModels.BEST:
        load_models_fn = load_best_model
    else:
        raise ValueError(f"load_models {load_models} is not supported!")

    inference_kwargs = cfg.get("inference_kwargs", {})
    logger.info(
        f"++ Running prediction of {preprocessed_images_dir} saving"
        f" into {prediction_dir} with inference kwargs {inference_kwargs},"
        f" {num_tta_transforms} tta transformations and {load_models} weights ++"
    )
    if case_ids is not None:
        logger.info(f"Running inference on provided case ids: {case_ids}")
    predict_dir(
        source_dir=preprocessed_images_dir,
        target_dir=prediction_dir,
        case_ids=case_ids,
        cfg=cfg,
        plan=plan,
        source_models=training_dir,
        num_models=None,
        num_tta_transforms=num_tta_transforms,
        model_fn=load_models_fn,
        restore=True,
        **inference_kwargs,
    )
    logger.info(f"++ Finished prediction, results located in {prediction_dir} ++")


@env_guard
def entrypoint_preprocess_for_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="Path to directory containing data.")
    parser.add_argument("task", type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument("model", type=str, help="model name, e.g. RetinaUNetV0")
    parser.add_argument(
        "fold",
        type=int,
        help="fold to use for prediction. -1 for consolidated",
    )
    parser.add_argument(
        "-npp",
        "--num_processes_preprocessing",
        type=int,
        default=2,
        required=False,
        help="Number of processes to use for preprocessing.",
    )
    args = parser.parse_args()
    data_dir = args.data
    task = args.task
    model = args.model
    fold = args.fold
    num_processes_preprocessing = args.num_processes_preprocessing

    # setup folders
    task_name = get_task(task, name=True)
    task_model_dir = Path(os.getenv("det_models"))
    training_dir = get_training_dir(task_model_dir / task_name / model, fold)

    preprocessed_dir: Path = data_dir / "preprocessed"
    preprocessed_dir.mkdir(exist_ok=True)
    _ = _preprocess(
        training_dir=training_dir,
        raw_splitted_dir=data_dir,
        preprocessed_dir=preprocessed_dir,
        num_processes=num_processes_preprocessing,
    )


@env_guard
def entrypoint_predict_with_task():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="Path to directory containing data.")
    parser.add_argument(
        "prediction",
        type=Path,
        help="Path to directory where predictions should be saved.",
    )
    parser.add_argument("task", type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument("model", type=str, help="model name, e.g. RetinaUNetV0")
    parser.add_argument(
        "fold",
        type=int,
        help="fold to use for prediction. -1 for consolidated",
    )
    parser.add_argument(
        "--skip_preprocessing",
        action="store_true",
        help="Skip preprocessing of data, data needs to be in preprocessed format already!",
    )
    parser.add_argument(
        "--load_models",
        type=str,
        help="Define model weights, one of all | last | best",
        default="all",
        required=False,
    )
    parser.add_argument(
        "-npp",
        "--num_processes_preprocessing",
        type=int,
        default=2,
        required=False,
        help="Number of processes to use for preprocessing.",
    )
    parser.add_argument(
        "-ntta",
        "--num_tta_transforms",
        type=int,
        default=None,
        help="number of tta transforms (per default most tta are chosen)",
        required=False,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=0,
        help="Batch size to use for inference. If 0, batch size from plan is used.",
        required=False,
    )
    parser.add_argument(
        "-o",
        "--overwrites",
        type=str,
        nargs="+",
        default=None,
        required=False,
        help=(
            "overwrites for config file. "
            "inference_kwargs can be used to add additional "
            "keyword arguments to inference."
        ),
    )
    args = parser.parse_args()
    data_dir = args.data
    prediction_dir = args.prediction
    task = args.task
    model = args.model
    fold = args.fold

    num_tta_transforms = args.num_tta_transforms
    batch_size = args.batch_size
    if batch_size == 0:
        batch_size = None
    load_models = LoadModels(args.load_models)
    num_processes_preprocessing = args.num_processes_preprocessing
    overwrites = args.overwrites

    skip_preprocessing = args.skip_preprocessing

    # setup folders
    task_name = get_task(task, name=True)
    task_model_dir = Path(os.getenv("det_models"))
    training_dir = get_training_dir(task_model_dir / task_name / model, fold)

    if skip_preprocessing:
        preprocessed_images_dir = data_dir
    else:
        preprocessed_dir: Path = data_dir / "preprocessed"
        preprocessed_dir.mkdir(exist_ok=True)
        data_identifier = _preprocess(
            training_dir=training_dir,
            raw_splitted_dir=data_dir,
            preprocessed_dir=preprocessed_dir,
            num_processes=num_processes_preprocessing,
        )
        preprocessed_images_dir = preprocessed_dir / data_identifier / "imagesTs"

    _predict(
        training_dir=training_dir,
        preprocessed_images_dir=preprocessed_images_dir,
        prediction_dir=prediction_dir,
        num_tta_transforms=num_tta_transforms,
        overwrites=overwrites,
        load_models=load_models,
        batch_size=batch_size,
        case_ids=None,
    )


@env_guard
def entrypoint_predict_with_folders():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="Path to directory containing data.")
    parser.add_argument(
        "prediction",
        type=Path,
        help="Path to directory where predictions should be saved.",
    )
    parser.add_argument("training", type=str, help="Directory to models weights, plan and config.")
    parser.add_argument(
        "--skip_preprocessing",
        action="store_true",
        help="Skip preprocessing of data, data needs to be in preprocessed format already!",
    )
    parser.add_argument(
        "--load_models",
        type=str,
        help="Define model weights, one of all | last | best",
        default="all",
        required=False,
    )
    parser.add_argument(
        "-npp",
        "--num_processes_preprocessing",
        type=int,
        default=2,
        required=False,
        help="Number of processes to use for preprocessing.",
    )
    parser.add_argument(
        "-ntta",
        "--num_tta_transforms",
        type=int,
        default=None,
        help="number of tta transforms (per default most tta are chosen)",
        required=False,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=0,
        help="Batch size to use for inference. If 0, batch size from plan is used.",
        required=False,
    )
    parser.add_argument(
        "-o",
        "--overwrites",
        type=str,
        nargs="+",
        default=None,
        required=False,
        help=(
            "overwrites for config file. "
            "inference_kwargs can be used to add additional "
            "keyword arguments to inference."
        ),
    )
    args = parser.parse_args()
    data_dir = args.data
    prediction_dir = args.prediction
    training_dir = args.training

    num_tta_transforms = args.num_tta_transforms
    batch_size = args.batch_size
    if batch_size == 0:
        batch_size = None
    load_models = LoadModels(args.load_models)
    num_processes_preprocessing = args.num_processes_preprocessing
    overwrites = args.overwrites

    skip_preprocessing = args.skip_preprocessing

    # setup folders
    if skip_preprocessing:
        preprocessed_images_dir = data_dir
    else:
        preprocessed_dir: Path = data_dir / "preprocessed"
        preprocessed_dir.mkdir(exist_ok=True)
        data_identifier = _preprocess(
            training_dir=training_dir,
            raw_splitted_dir=data_dir,
            preprocessed_dir=preprocessed_dir,
            num_processes=num_processes_preprocessing,
        )
        preprocessed_images_dir = preprocessed_dir / data_identifier / "imagesTs"

    _predict(
        training_dir=training_dir,
        preprocessed_images_dir=preprocessed_images_dir,
        prediction_dir=prediction_dir,
        num_tta_transforms=num_tta_transforms,
        overwrites=overwrites,
        load_models=load_models,
        batch_size=batch_size,
        case_ids=None,
    )


@env_guard
def entrypoint_predict_test_split():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="Path to directory containing data.")
    parser.add_argument(
        "prediction",
        type=Path,
        help="Path to directory where predictions should be saved.",
    )
    parser.add_argument("task", type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument("model", type=str, help="model name, e.g. RetinaUNetV0")
    parser.add_argument(
        "fold",
        type=int,
        help="fold to use for prediction. -1 for consolidated",
    )
    parser.add_argument(
        "--skip_preprocessing",
        action="store_true",
        help="Skip preprocessing of data, data needs to be in preprocessed format already!",
    )
    parser.add_argument(
        "--load_models",
        type=str,
        help="Define model weights, one of all | last | best",
        default="all",
        required=False,
    )
    parser.add_argument(
        "-npp",
        "--num_processes_preprocessing",
        type=int,
        default=2,
        required=False,
        help="Number of processes to use for preprocessing.",
    )
    parser.add_argument(
        "-ntta",
        "--num_tta_transforms",
        type=int,
        default=None,
        help="number of tta transforms (per default most tta are chosen)",
        required=False,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=0,
        help="Batch size to use for inference. If 0, batch size from plan is used.",
        required=False,
    )
    parser.add_argument(
        "-o",
        "--overwrites",
        type=str,
        nargs="+",
        default=None,
        required=False,
        help=(
            "overwrites for config file. "
            "inference_kwargs can be used to add additional "
            "keyword arguments to inference."
        ),
    )
    args = parser.parse_args()
    task = args.task
    model = args.model
    fold = args.fold
    if fold == -1:
        raise ValueError("Fold 'consolidated' is not compatible with test split inference.")

    num_tta_transforms = args.num_tta_transforms
    batch_size = args.batch_size
    if batch_size == 0:
        batch_size = None
    load_models = LoadModels(args.load_models)
    if load_models == LoadModels.ALL:
        raise ValueError("Load all models is not compatible with test split inference.")
    overwrites = args.overwrites

    # setup folders
    task_name = get_task(task, name=True)
    task_model_dir = Path(os.getenv("det_models"))
    training_dir = get_training_dir(task_model_dir / task_name / model, fold)
    prediction_dir = training_dir / "test_predictions"

    # determine preprocessed data
    plan_inference_path = training_dir / "plan_inference.pkl"
    if not plan_inference_path.is_file():
        raise RuntimeError(
            f"Expected {plan_inference_path} to contain the plan for "
            "running inference. Either run nndet_consolidate to predict "
            "ensembles or nndet_sweep for single fold models."
        )
    plan = load_pickle(plan_inference_path)
    preprocessed_images_dir = Path(os.getenv("det_data")) / "preprocessed" / plan["data_identifier"] / "imagesTr"

    # determine case ids
    splits_path = training_dir / "splits.pkl"
    if not splits_path.is_file():
        raise RuntimeError(f"Expected {splits_path} to contain the splits for " "running inference.")
    case_ids = load_pickle(splits_path)[fold]["test"]

    _predict(
        training_dir=training_dir,
        preprocessed_images_dir=preprocessed_images_dir,
        prediction_dir=prediction_dir,
        num_tta_transforms=num_tta_transforms,
        overwrites=overwrites,
        load_models=load_models,
        batch_size=batch_size,
        case_ids=case_ids,
    )


if __name__ == "__main__":
    entrypoint_predict_with_task()
