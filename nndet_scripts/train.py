# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib
import os
import socket
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Union

import pytorch_lightning as pl
import torch
from hydra import initialize_config_module
from loguru import logger
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, Logger, MLFlowLogger, TensorBoardLogger
from pytorch_lightning.plugins.precision import MixedPrecisionPlugin

import nndet
from nndet.eval.registry import (
    evaluate_box_dir,
    evaluate_box_dir_bootstrap,
    evaluate_case_dir,
)
from nndet.inference.helper import extract_results
from nndet.io.datamodule.module import PtDatamodule as Datamodule
from nndet.io.load import load_json, load_pickle, load_yaml, save_json, save_pickle
from nndet.io.paths import get_task, get_training_dir
from nndet.ptmodule import MODULE_REGISTRY
from nndet.utils.check import env_guard
from nndet.utils.config import compose, load_dataset_info
from nndet.utils.info import (
    ModelSummary,
    create_debug_plan,
    flatten_mapping,
    host_and_env_info,
    log_git,
    write_requirements,
)


@env_guard
def train() -> None:
    """
    Training entry
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument("train_config", type=str, help="specify model config to use for training")
    parser.add_argument(
        "fold",
        type=int,
        help="fold to train",
    )
    parser.add_argument(
        "-o",
        "--overwrites",
        type=str,
        nargs="+",
        help="overwrites for config file",
        required=False,
    )
    parser.add_argument(
        "-ct",
        "--continue_training",
        help="Continue training from last checkpoint",
        action="store_true",
    )
    parser.add_argument(
        "--sweep",
        help="Run empirical parameter optimization",
        action="store_true",
    )
    parser.add_argument(
        "--log_net",
        help="Log network structure in console",
        action="store_true",
    )
    parser.add_argument(
        "--log_aug",
        help="Log augmentation in console",
        action="store_true",
    )
    parser.add_argument(
        "-tl",
        "--transfer_learning",
        help=(
            "If this option is acivated, the training script will look for a "
            "`model_transfer` checkpoint in the training directory and use the "
            "weights to initialise the model. It is not possible to use this "
            "command in conjunction with continue training. Simply use continue "
            "training option to continue transfer learning experiments. Make sure "
            "to place other checkpoints like `model_last` or `model_best` in a "
            "different directory."
        ),
        action="store_true",
    )

    args = parser.parse_args()

    task = args.task
    train_config = args.train_config
    fold = args.fold

    ov = args.overwrites
    continue_training = args.continue_training
    transfer_learning = args.transfer_learning
    do_sweep = args.sweep
    log_net = args.log_net
    log_aug = args.log_aug

    _train(
        task=task,
        train_config=train_config,
        fold=fold,
        ov=ov,
        continue_training=continue_training,
        transfer_learning=transfer_learning,
        do_sweep=do_sweep,
        log_net=log_net,
        log_aug=log_aug,
    )


@env_guard
def sweep() -> None:
    """
    Sweep entry
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument(
        "model",
        type=str,
        help="full name of experiment to sweep e.g. RetinaUNetV0_D3V001_3d",
    )
    parser.add_argument("fold", type=int, help="experiment fold")
    parser.add_argument(
        "--no_pred",
        help="Turn of model prediction",
        action="store_true",
    )
    args = parser.parse_args()
    task = args.task
    model = args.model
    fold = args.fold
    run_prediction = bool(not args.no_pred)
    _sweep(
        task=task,
        model=model,
        fold=fold,
        run_prediction=run_prediction,
    )


@env_guard
def evaluate() -> None:
    """
    Evaluation entry

    seg, instances are not supported yet
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument("model", type=str, help="model name, e.g. RetinaUNetV0_D3V001_3d")
    parser.add_argument("fold", type=int, help="fold, -1 => consolidated")

    parser.add_argument(
        "--test",
        help="Evaluate test predictions -> uses different folder",
        action="store_true",
    )
    parser.add_argument("--case", help="Run Case Evaluation", action="store_true")
    parser.add_argument("--boxes", help="Run Box Evaluation", action="store_true")
    parser.add_argument("--analyze_boxes", help="Analyze Box Results", action="store_true")
    parser.add_argument(
        "--eval_preprocessed",
        help="Additionally run evaluation on preprocessed data",
        action="store_true",
    )
    parser.add_argument(
        "--bootstrapping",
        help="Additionally run evaluation with bootstrapping",
        action="store_true",
    )

    args = parser.parse_args()
    model: str = args.model
    fold: int = args.fold
    task: str = args.task
    test: bool = args.test
    eval_preprocessed: bool = args.eval_preprocessed

    do_boxes_eval: bool = args.boxes
    do_case_eval: bool = args.case
    do_bootstrapping: bool = args.bootstrapping

    _evaluate_task(
        task=task,
        model=model,
        fold=fold,
        test=test,
        preprocessed=eval_preprocessed,
        do_boxes_eval=do_boxes_eval,
        do_case_eval=do_case_eval,
        do_bootstrapping=do_bootstrapping,
    )


@env_guard
def evaluate_with_folders() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_dir", type=Path, help="path to directory with predictions")
    parser.add_argument("gt_dir", type=Path, help="path to directory with ground truth data")
    parser.add_argument("save_dir", type=Path, help="path to directory where results should be saved")
    parser.add_argument(
        "data_cfg_path",
        type=Path,
        help="path to dataset.yaml or dataset.json file of data",
    )

    parser.add_argument("--case", help="Run Case Evaluation", action="store_true")
    parser.add_argument("--boxes", help="Run Box Evaluation", action="store_true")
    parser.add_argument("--analyze_boxes", help="Analyze Box Results", action="store_true")
    parser.add_argument(
        "--bootstrapping",
        help="Additionally run evaluation with bootstrapping",
        action="store_true",
    )

    args = parser.parse_args()
    pred_dir: Path = args.pred_dir
    gt_dir: Path = args.gt_dir
    save_dir: Path = args.save_dir
    data_cfg_path: Path = args.data_cfg_path

    if not data_cfg_path.is_file():
        raise ValueError(f"data cfg path needs to be a file, {data_cfg_path} is not.")
    if data_cfg_path.suffix == ".json":
        data_cfg = load_json(data_cfg_path)
    else:
        data_cfg = load_yaml(data_cfg_path)

    do_case_eval: bool = args.case
    do_boxes_eval: bool = args.boxes
    do_bootstrapping: bool = args.bootstrapping

    # logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<level>{level} {message}</level>",
        level="INFO",
        colorize=True,
    )
    logger.add(save_dir / "evaluation.log", level="INFO")
    current_time = datetime.now()
    current_time_str = current_time.strftime("%d/%m/%Y %H:%M:%S")
    logger.info(f"+++ Running evaluation {current_time_str} +++")

    _evaluate(
        data_cfg=data_cfg,
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        save_dir=save_dir,
        do_case_eval=do_case_eval,
        do_boxes_eval=do_boxes_eval,
        do_bootstrapping=do_bootstrapping,
    )


def init_train_dir(cfg: dict, fold: int) -> Path:
    """
    Initialize training directory and make it the current working directory
    """
    # determine folder for experiment
    output_dir = Path(os.getenv("det_models")) / str(cfg.task) / str(cfg.exp.id) / f"fold{fold}"

    if output_dir.is_dir():
        print(f"Found existing folder {output_dir}, this run will might overwrite the results inside that folder")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(str(output_dir))
    return output_dir


def get_pl_logger(cfg: dict, fold: int) -> Union[Logger, bool]:
    """
    Instantiate a logger to monitor metrics/losses during training

    Args:
        cfg: config
            'logger': define logger type

    Returns:
        LightningLoggerBase: Instantiated logger
    """
    logger_name = os.getenv("det_logger", "mlflow").lower()
    save_dir = os.getenv("det_logging", None)

    pl_logger = [
        CSVLogger(
            save_dir="./logging",
            name="csv",
            version=None,
        ),
    ]

    # logger not defined
    if logger_name == "none":
        pass
    elif logger_name == "mlflow":
        if save_dir is not None:
            save_dir = Path(save_dir)
            if not save_dir.name == "mlruns":
                save_dir = save_dir / "mlruns"
        else:
            save_dir = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")

        run_name = cfg["exp"]["id"]
        tags = {
            "host": socket.gethostname(),
            "fold": fold,
            "task": cfg["task"],
            "job_id": os.getenv("LSB_JOBID", "no_id"),
            "mlflow.runName": run_name,
        }
        mlflow_logger = MLFlowLogger(
            experiment_name=cfg["task"],
            tags=tags,
            save_dir=save_dir,
        )
        if (ml_exp := mlflow_logger._mlflow_client.get_experiment_by_name(cfg["task"])) is not None:
            exp_id = ml_exp.experiment_id
            runs = mlflow_logger._mlflow_client.search_runs([exp_id], filter_string=f'tag.mlflow.runName="{run_name}"')
            if len(runs) > 0:
                mlflow_logger.tags["mlflow.parentRunId"] = runs[-1].info.run_id
        pl_logger.append(mlflow_logger)
    elif logger_name == "tensorboard":
        if save_dir is not None:
            save_dir = Path(save_dir) / "tbruns" / cfg["task"]
            name = f"{cfg['exp']['id']}_fold{fold}"
        else:
            save_dir = "./logging"
            name = "tboard"

        tb_logger = TensorBoardLogger(
            save_dir=save_dir,
            name=name,
            version=None,
            default_hp_metric=True,
        )
        pl_logger.append(tb_logger)
    else:
        raise ValueError(f"Logger {logger_name} is not supported!")
    return pl_logger


def _train(
    task: str,
    train_config: str,
    fold: int,
    ov: List[str],
    do_sweep: bool,
    continue_training: bool,
    transfer_learning: bool,
    log_net: bool = False,
    log_aug: bool = False,
):
    """
    Run training

    Args:
        task: task to run training for
        train_config: name of config to use for training
        fold: number of fold to train
        ov: overwrites for config manager
        do_sweep: determine best emprical parameters for run
        continue_training: continue training from last model checkpoint
        transfer_learning: init model with weights from other training
        log_net: print the network architecture
        log_aug: print the augmentation pipeline
    """
    print(f"Overwrites: {ov}")
    ov = [] if ov is None else ov
    if any("train=" in o for o in ov):
        raise ValueError("Can not overwrite train config via overwrites anymore, use train_config parameter instead.")
    ov.insert(0, f"train={train_config}")

    initialize_config_module(config_module="nndet.conf", version_base="1.1")
    cfg = compose(task, "config.yaml", overrides=ov)

    train_dir = init_train_dir(cfg, fold=fold)
    pl_logger = get_pl_logger(cfg, fold=fold)
    params = {
        "module": cfg["module"],
        "plan": cfg["plan"],
        "aug_name": cfg["augment_cfg"]["name"],
        "aug_transforms": cfg["augment_cfg"]["transforms"],
        **flatten_mapping({"model": OmegaConf.to_container(cfg["model_cfg"], resolve=True)}),
        **flatten_mapping({"trainer": OmegaConf.to_container(cfg["trainer_cfg"], resolve=True)}),
    }
    for _pl_logger in pl_logger:
        _pl_logger.log_hyperparams(params)

    logger.remove()
    logger.add(
        sys.stdout,
        format="<level>{level}</level>: {message}",
        level="INFO",
        colorize=True,
    )
    log_file = Path(os.getcwd()) / "train.log"
    logger.add(log_file, level="INFO")
    current_time = datetime.now()
    current_time_str = current_time.strftime("%d/%m/%Y %H:%M:%S")
    logger.info(f"+++ Running train {current_time_str} +++")
    logger.info(f"Log file at {log_file}")
    logger.info(f"Training with overwrites: {ov}")

    meta_data = {}
    meta_data["torch_version"] = str(torch.__version__)
    meta_data["date"] = str(datetime.now())
    meta_data["git"] = log_git(nndet.__path__[0], repo_name="nndet")
    meta_data["host"] = socket.gethostname()
    meta_data["job_id"] = os.getenv("LSB_JOBID", "no_id")
    meta_data["overwrites"] = str(ov)
    save_json(meta_data, "./meta.json")
    _ = write_requirements(train_dir)

    plan_path = Path(os.getenv("det_data")) / cfg["task"] / "preprocessed" / f"{cfg['plan']}.pkl"
    plan = load_pickle(plan_path)
    data_dir = Path(os.getenv("det_data")) / cfg["task"] / "preprocessed" / plan["data_identifier"] / "imagesTr"

    # initiate module
    module = MODULE_REGISTRY[cfg["module"]](
        model_cfg=OmegaConf.to_container(cfg["model_cfg"], resolve=True),
        trainer_cfg=OmegaConf.to_container(cfg["trainer_cfg"], resolve=True),
        accelerator_cfg=OmegaConf.to_container(cfg["accelerator_cfg"], resolve=True),
        plan=plan,
    )

    # setup io
    datamodule = Datamodule(
        io_cfg=OmegaConf.to_container(cfg["io_cfg"], resolve=True),
        augment_cfg=OmegaConf.to_container(cfg["augment_cfg"], resolve=True),
        plan=plan,
        data_dir=data_dir,
        fold=fold,
        use_box_io=module.use_box_io(),
        log_aug=log_aug,
    )
    plan["patch_size"] = list(datamodule.patch_size)
    plan["batch_size"] = int(datamodule.batch_size)

    # callbacks
    callbacks = []
    checkpoint_cb = ModelCheckpoint(
        dirpath=train_dir,
        filename="model_best",
        save_last=True,
        save_top_k=cfg["trainer_cfg"].get("save_top_k", 1),
        monitor=cfg["trainer_cfg"]["monitor_key"],
        mode=cfg["trainer_cfg"]["monitor_mode"],
        enable_version_counter=False,
    )
    checkpoint_cb.CHECKPOINT_NAME_LAST = "model_last"
    callbacks.append(checkpoint_cb)
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    OmegaConf.save(cfg, str(Path(os.getcwd()) / "config.yaml"))
    OmegaConf.save(cfg, str(Path(os.getcwd()) / "config_resolved.yaml"), resolve=True)
    save_pickle(plan, train_dir / "plan.pkl")  # backup plan
    save_json(create_debug_plan(plan), "./plan_debug.json")  # easy read backup
    splits = load_pickle(Path(os.getenv("det_data")) / cfg["task"] / "preprocessed" / datamodule.splits_file)
    save_pickle(splits, train_dir / "splits.pkl")

    trainer_kwargs = {}
    if continue_training:
        _path = train_dir / "model_last.ckpt"
        logger.info(f"Continue training -> loading checkpoint: {_path}")
        trainer_kwargs["resume_from_checkpoint"] = _path
    if transfer_learning:
        _path = train_dir / "model_transfer.ckpt"
        logger.info(f"Performing transfer learning -> loading model weights: {_path}")
        if continue_training:
            _s = "Found continue training and transfer learning, only one can be activated at the same time!"
            logger.error(_s)
            raise RuntimeError(_s)
        else:
            if not _path.is_file():
                _s = f"Transfer learning active, expected {_path} to exist."
                logger.error(_s)
                raise RuntimeError(_s)
            module.load_state_dict(torch.load(_path)["state_dict"], strict=True)

    num_gpus = cfg["accelerator_cfg"]["gpus"]
    logger.info(f"Using {num_gpus} GPUs for training")

    plugins = []
    if p := cfg["trainer_cfg"].get("plugins", None):
        plugins.append(p)
    logger.info(f"Using {plugins} plugins for training")

    callbacks.append(ModelSummary(max_depth=10, log_net=log_net))
    if bool(int(os.getenv("det_verbose", 1))):
        enable_progress_bar = True
        callbacks.append(TQDMProgressBar())
    else:
        enable_progress_bar = False

    if "terminate_on_nan" in cfg["accelerator_cfg"]:
        detect_anomaly = cfg["accelerator_cfg"]["terminate_on_nan"]
    elif "detect_anomaly" in cfg["accelerator_cfg"]:
        detect_anomaly = cfg["accelerator_cfg"]["detect_anomaly"]
    else:
        detect_anomaly = False

    if cfg["accelerator_cfg"]["precision"] == "16-mixed":
        logger.info("Using mixed precision training: '16-mixed'")
        device = "cuda" if num_gpus > 0 else "cpu"
        scaler = torch.cuda.amp.GradScaler(init_scale=8192.0)
        precision_plugin = MixedPrecisionPlugin(
            precision=cfg["accelerator_cfg"]["precision"],
            device=device,
            scaler=scaler,
        )
        plugins.append(precision_plugin)
        precision = None
    else:
        precision = cfg["accelerator_cfg"]["precision"]

    logger.info(
        "Running experiment with accelerator benchmark: "
        f"{cfg['accelerator_cfg']['benchmark']} and "
        f"deterministic: {cfg['accelerator_cfg']['deterministic']}"
    )

    trainer = pl.Trainer(
        accelerator=cfg["accelerator_cfg"]["accelerator"],
        devices=list(range(num_gpus)) if num_gpus > 1 else num_gpus,
        precision=precision,
        benchmark=cfg["accelerator_cfg"]["benchmark"],
        deterministic=cfg["accelerator_cfg"]["deterministic"],
        callbacks=callbacks,
        logger=pl_logger,
        max_epochs=module.max_epochs,
        num_sanity_val_steps=10,
        plugins=plugins,
        detect_anomaly=detect_anomaly,
        enable_model_summary=False,
        enable_progress_bar=enable_progress_bar,
        **trainer_kwargs,
    )

    train_start = time.time()
    trainer.fit(module, datamodule=datamodule)
    train_end = time.time()
    train_time = train_end - train_start

    run_info = host_and_env_info()
    run_info["train_s"] = train_time
    run_info["train_h"] = train_time / 3600
    if do_sweep:
        case_ids = splits[fold]["val"]
        if "debug" in cfg["trainer_cfg"] and "num_cases_val" in cfg["trainer_cfg"]["debug"]:
            logger.warning("[!!!] Detected debug mode for sweep using reduced set of cases")
            case_ids = case_ids[: cfg["trainer_cfg"]["debug"]["num_cases_val"]]

        sweep_start = time.time()
        inference_plan = module.sweep(
            cfg=OmegaConf.to_container(cfg, resolve=True),
            save_dir=train_dir,
            train_data_dir=data_dir,
            case_ids=case_ids,
            run_prediction=True,
        )
        sweep_end = time.time()
        sweep_time = sweep_end - sweep_start
        run_info["sweep_s"] = sweep_time
        run_info["sweep_h"] = sweep_time / 3600

        plan["inference_plan"] = inference_plan
        save_pickle(plan, train_dir / "plan_inference.pkl")

        eval_start = time.time()
        ensembler_cls = module.get_ensembler_cls(dim=plan["network_dim"])
        for restore in [True, False]:
            target_dir = train_dir / "val_predictions" if restore else train_dir / "val_predictions_preprocessed"
            extract_results(
                source_dir=train_dir / "sweep_predictions",
                target_dir=target_dir,
                ensembler_cls=ensembler_cls,
                restore=restore,
                **inference_plan,
            )
        _evaluate_task(
            task=cfg["task"],
            model=cfg["exp"]["id"],
            fold=fold,
            test=False,
            preprocessed=True,
            do_case_eval=(module.requires_case_eval and (cfg["data"]["target_class"] is not None)),
            do_boxes_eval=module.requires_box_eval(),
        )
        eval_end = time.time()
        eval_time = eval_end - eval_start
        run_info["eval_s"] = eval_time
        run_info["eval_h"] = eval_time / 3600
    save_json(run_info, "./run_info.json")


def _sweep(
    task: str,
    model: str,
    fold: int,
    run_prediction: bool,
):
    """
    Determine best postprocessing parameters for a trained model

    Args:
        task: current task
        model: full name of the model run determine empricial parameters for
            e.g. RetinaUNetV001_D3V001_3d
        fold: current fold
    """
    nndet_model_dir = Path(os.getenv("det_models"))
    task = get_task(task, name=True, models=True)
    train_dir = nndet_model_dir / task / model / f"fold{fold}"

    cfg = OmegaConf.load(str(train_dir / "config.yaml"))
    os.chdir(str(train_dir))

    for imp in cfg.get("additional_imports", []):
        print(f"Additional import found {imp}")
        importlib.import_module(imp)

    if cfg["exp"]["id"] != model:
        raise ValueError("Config and model name do not match! " f"Found {cfg['exp']['id']} and {model} in cfg & model")

    logger.remove()
    logger.add(
        sys.stdout,
        format="<level>{level}</level>: {message}",
        level="INFO",
        colorize=True,
    )
    log_file = Path(os.getcwd()) / "sweep.log"
    logger.add(log_file, level="INFO")
    current_time = datetime.now()
    current_time_str = current_time.strftime("%d/%m/%Y %H:%M:%S")
    logger.info(f"+++ Running sweep {current_time_str} +++")
    logger.info(f"Log file at {log_file}")

    plan = load_pickle(train_dir / "plan.pkl")
    data_dir = Path(os.getenv("det_data")) / cfg["task"] / "preprocessed" / plan["data_identifier"] / "imagesTr"

    module = MODULE_REGISTRY[cfg["module"]](
        model_cfg=OmegaConf.to_container(cfg["model_cfg"], resolve=True),
        trainer_cfg=OmegaConf.to_container(cfg["trainer_cfg"], resolve=True),
        plan=plan,
    )

    splits = load_pickle(train_dir / "splits.pkl")
    case_ids = splits[fold]["val"]

    if "debug" in cfg["trainer_cfg"] and "num_cases_val" in cfg["trainer_cfg"]["debug"]:
        logger.warning("Detected debug mode for sweep using reduced set of cases!")
        case_ids = case_ids[: cfg["trainer_cfg"]["debug"]["num_cases_val"]]
    # case_ids = case_ids[:10]

    inference_plan = module.sweep(
        cfg=OmegaConf.to_container(cfg, resolve=True),
        save_dir=train_dir,
        train_data_dir=data_dir,
        case_ids=case_ids,
        run_prediction=run_prediction,
    )

    plan["inference_plan"] = inference_plan
    save_pickle(plan, train_dir / "plan_inference.pkl")

    ensembler_cls = module.get_ensembler_cls(dim=plan["network_dim"])
    for restore in [True, False]:
        target_dir = train_dir / "val_predictions" if restore else train_dir / "val_predictions_preprocessed"
        extract_results(
            source_dir=train_dir / "sweep_predictions",
            target_dir=target_dir,
            ensembler_cls=ensembler_cls,
            restore=restore,
            **inference_plan,
        )

    _evaluate_task(
        task=cfg["task"],
        model=model,
        fold=fold,
        test=False,
        preprocessed=True,
        do_case_eval=(module.requires_case_eval and (cfg["data"]["target_class"] is not None)),
        do_boxes_eval=module.requires_box_eval(),
    )


def _evaluate_task(
    task: str,
    model: str,
    fold: int,
    test: bool = False,
    preprocessed: bool = False,
    do_case_eval: bool = False,
    do_boxes_eval: bool = False,
    do_bootstrapping: bool = False,
) -> None:
    """
    Run evaluation on task (old behavior of _evaluate function in V0.1)

    Args:
        task: current task
        model: full name of the model run determine empricial parameters for
            e.g. RetinaUNetV001_D3V001_3d
        fold: current fold
        test: use test split
        preprocessed: indicate if predictions and labels are preprocessed
        do_case_eval: evaluate patient metrics
        do_boxes_eval: perform box evaluation
        do_bootstrapping: run bootstrapping for evaluation
    """
    # prepare paths
    task = get_task(task, name=True)
    model_dir = Path(os.getenv("det_models")) / task / model

    data_dir_task = Path(os.getenv("det_data")) / task
    data_cfg = load_dataset_info(data_dir_task)

    training_dir = get_training_dir(model_dir, fold)
    prefix = "test" if test else "val"

    # logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<level>{level} {message}</level>",
        level="INFO",
        colorize=True,
    )
    logger.add(training_dir / "evaluation.log", level="INFO")
    current_time = datetime.now()
    current_time_str = current_time.strftime("%d/%m/%Y %H:%M:%S")
    logger.info(f"+++ Running evaluation {current_time_str} +++")

    # prepare paths
    modes = [True]
    if not test and preprocessed:
        modes.append(False)
    if test and preprocessed:
        logger.warning("Evaluation of preprocessed data only supported on validation data.")

    for restore in modes:
        if restore:
            pred_dir_name = f"{prefix}_predictions"
            gt_dir_name = "labelsTs" if test else "labelsTr"
            gt_dir = data_dir_task / "preprocessed" / gt_dir_name
        else:
            plan = load_pickle(training_dir / "plan.pkl")
            pred_dir_name = f"{prefix}_predictions_preprocessed"
            gt_dir = data_dir_task / "preprocessed" / plan["data_identifier"] / "labelsTr"

        pred_dir = training_dir / pred_dir_name
        save_dir = training_dir / f"{prefix}_results" if restore else training_dir / f"{prefix}_results_preprocessed"

        _evaluate(
            data_cfg=data_cfg,
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            save_dir=save_dir,
            do_case_eval=do_case_eval,
            do_boxes_eval=do_boxes_eval,
            do_bootstrapping=do_bootstrapping,
        )


@env_guard
def _evaluate(
    data_cfg: dict,
    pred_dir: os.PathLike,
    gt_dir: os.PathLike,
    save_dir: os.PathLike,
    do_case_eval: bool = False,
    do_boxes_eval: bool = False,
    do_bootstrapping: bool = False,
) -> None:
    """
    Run evaluation

    Args:
        prediction_dir: path to directory containing the predictions
        label_dir: path to directory containing the ground truth labels
        save_dir: path to directory where results should be saved
        do_case_eval: evaluate patient metrics
        do_boxes_eval: perform box evaluation
        do_bootstrapping: run bootstrapping for box evaluation
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # handle case level evaluation
    if do_case_eval:
        logger.info("Computing case metrics")
        scores, curves = evaluate_case_dir(
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            classes=list(data_cfg["labels"].keys()),
            target_class=data_cfg["target_class"],
        )
        save_json(
            {str(key): str(item) for key, item in scores.items()},
            save_dir / "results_case.json",
        )
        save_pickle({"scores": scores, "curves": curves}, save_dir / "results_case.pkl")

    # handle box level evaluation
    if do_boxes_eval:
        logger.info("Computing box metrics")
        scores, curves = evaluate_box_dir(
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            classes=list(data_cfg["labels"].keys()),
            save_dir=save_dir / "boxes",
        )
        save_json(
            {str(key): str(item) for key, item in scores.items()},
            save_dir / "results_boxes.json",
        )
        save_pickle(
            {
                "scores": scores,
                "curves": curves,
            },
            save_dir / "results_boxes.pkl",
        )

        # optionally run results with bootstrapping
        if do_bootstrapping:
            logger.info("Computing box metrics with bootstrapping")
            iqr_scores, scores_boot, curves_boot = evaluate_box_dir_bootstrap(
                pred_dir=pred_dir,
                gt_dir=gt_dir,
                classes=list(data_cfg["labels"].keys()),
                iterations=1000,
                iqr=0.95,
                seed=0,
            )
            save_json(iqr_scores, save_dir / "results_boxes_iqr_boot.json")
            scores_boot_dict_list = defaultdict(list)
            for _scores_boot in scores_boot:
                for _key, _value in _scores_boot.items():
                    scores_boot_dict_list[_key].append(_value)
            save_json(scores_boot_dict_list, save_dir / "results_boxes_boot.json")
            save_pickle(
                {
                    "iqr_scores": iqr_scores,
                    "scores": scores_boot,
                    "curves": curves_boot,
                },
                save_dir / "results_boxes_boot.pkl",
            )


if __name__ == "__main__":
    train()
