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
import importlib
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Union

import pytorch_lightning as pl
import torch
from hydra import initialize_config_module
from loguru import logger
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import (
    LightningLoggerBase,
    MLFlowLogger,
    TensorBoardLogger,
)

import nndet
from nndet.evaluator.registry import (
    evaluate_box_dir,
    evaluate_case_dir,
    evaluate_mask_dir,
    evaluate_seg_dir,
    save_metric_output,
)
from nndet.inference.helper import extract_results
from nndet.io.datamodule.module import PtDatamodule as Datamodule
from nndet.io.load import load_pickle, save_json, save_pickle
from nndet.io.paths import get_task, get_training_dir
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.optimizer.amp import ExposedNativeMixedPrecisionPlugin
from nndet.utils.analysis import run_analysis_suite
from nndet.utils.check import env_guard
from nndet.utils.config import compose, load_dataset_info
from nndet.utils.info import (
    ModelSummary,
    create_debug_plan,
    flatten_mapping,
    host_and_env_info,
    log_git,
)


@env_guard
def train():
    """
    Training entry
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument(
        "-o",
        "--overwrites",
        type=str,
        nargs="+",
        help="overwrites for config file",
        required=False,
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

    args = parser.parse_args()
    task = args.task
    ov = args.overwrites
    do_sweep = args.sweep
    log_net = args.log_net
    log_aug = args.log_aug

    _train(
        task=task,
        ov=ov,
        do_sweep=do_sweep,
        log_net=log_net,
        log_aug=log_aug,
    )


@env_guard
def sweep():
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
def evaluate():
    """
    Evaluation entry

    seg, instances are not supported yet
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument(
        "model", type=str, help="model name, e.g. RetinaUNetV0_D3V001_3d"
    )
    parser.add_argument("fold", type=int, help="fold, -1 => consolidated")

    parser.add_argument(
        "--test",
        help="Evaluate test predictions -> uses different folder",
        action="store_true",
    )
    parser.add_argument("--case", help="Run Case Evaluation", action="store_true")
    parser.add_argument("--boxes", help="Run Box Evaluation", action="store_true")
    parser.add_argument(
        "--seg", help="Run Semantic Segmentation Evaluation", action="store_true"
    )
    parser.add_argument("--masks", help="Run Mask Evaluation", action="store_true")
    parser.add_argument(
        "--analyze_boxes", help="Analyze Box Results", action="store_true"
    )
    parser.add_argument(
        "--eval_preprocessed",
        help="Additionally run evaluation on preprocessed data",
        action="store_true",
    )

    args = parser.parse_args()
    model = args.model
    fold = args.fold
    task = args.task
    test = args.test

    do_boxes_eval = args.boxes
    do_case_eval = args.case
    do_seg_eval = args.seg
    do_masks_eval = args.masks

    do_analyze_boxes = args.analyze_boxes

    eval_preprocessed = args.eval_preprocessed

    _evaluate(
        task=task,
        model=model,
        fold=fold,
        test=test,
        do_boxes_eval=do_boxes_eval,
        do_case_eval=do_case_eval,
        do_seg_eval=do_seg_eval,
        do_masks_eval=do_masks_eval,
        do_analyze_boxes=do_analyze_boxes,
        eval_preprocessed=eval_preprocessed,
    )


def init_train_dir(cfg) -> Path:
    """
    Initialize training directory and make it the current working directory
    """
    # determine folder for experiment
    output_dir = (
        Path(cfg.host.parent_results)
        / str(cfg.task)
        / str(cfg.exp.id)
        / f"fold{cfg.exp.fold}"
    )

    if cfg["exec"]["mode"].lower() == "overwrite":
        if output_dir.is_dir():
            print(
                f"Found existing folder {output_dir}, this run will overwrite "
                f"the results inside that folder"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        if not output_dir.is_dir():
            raise ValueError(
                f"{output_dir} is not a valid training dir and thus can not be resumed"
            )
    os.chdir(str(output_dir))
    return output_dir


def get_pl_logger(cfg: dict) -> Union[LightningLoggerBase, bool]:
    """
    Instantiate a logger to monitor metrics/losses during training

    Args:
        cfg: config
            'logger': define logger type

    Returns:
        LightningLoggerBase: Instantiated logger
    """
    logger_name = cfg["exec"].get("logger", "mlflow")
    if isinstance(logger_name, str):
        logger_name = logger_name.lower()

    pl_logger = False
    save_dir = os.getenv("det_logging", None)

    # logger not defined
    if logger_name.lower() == "none":
        return pl_logger

    if logger_name == "mlflow":
        if save_dir is not None:
            save_dir = Path(save_dir)
            if not save_dir.name == "mlruns":
                save_dir = save_dir / "mlruns"
        else:
            save_dir = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")

        run_name = cfg["exp"]["id"]
        tags = {
            "host": socket.gethostname(),
            "fold": cfg["exp"]["fold"],
            "task": cfg["task"],
            "job_id": os.getenv("LSB_JOBID", "no_id"),
            "mlflow.runName": run_name,
        }
        pl_logger = MLFlowLogger(
            experiment_name=cfg["task"],
            tags=tags,
            save_dir=save_dir,
        )
        if (
            ml_exp := pl_logger._mlflow_client.get_experiment_by_name(cfg["task"])
        ) is not None:
            exp_id = ml_exp.experiment_id
            runs = pl_logger._mlflow_client.search_runs(
                [exp_id], filter_string=f'tag.mlflow.runName="{run_name}"'
            )
            if len(runs) > 0:
                pl_logger.tags["mlflow.parentRunId"] = runs[-1].info.run_id
    elif logger_name == "tensorboard":
        if save_dir is not None:
            save_dir = Path(save_dir) / "tbruns" / cfg["task"]
        else:
            save_dir = "./logging"

        pl_logger = TensorBoardLogger(
            save_dir=save_dir,
            name=f"{cfg['exp']['id']}_fold{cfg['exp']['fold']}",
            default_hp_metric=True,
        )
    return pl_logger


def _train(
    task: str,
    ov: List[str],
    do_sweep: bool,
    log_net: bool = False,
    log_aug: bool = False,
):
    """
    Run training

    Args:
        task: task to run training for
        ov: overwrites for config manager
        do_sweep: determine best emprical parameters for run
    """
    print(f"Overwrites: {ov}")
    initialize_config_module(config_module="nndet.conf", version_base="1.1")
    cfg = compose(task, "config.yaml", overrides=ov if ov is not None else [])

    assert cfg.host.parent_data is not None, "Parent data can not be None"
    assert cfg.host.parent_results is not None, "Output dir can not be None"

    train_dir = init_train_dir(cfg)
    pl_logger = get_pl_logger(cfg)
    if pl_logger:
        params = {
            "module": cfg["module"],
            "plan": cfg["plan"],
            "aug_name": cfg["augment_cfg"]["name"],
            "aug_transforms": cfg["augment_cfg"]["transforms"],
            **flatten_mapping(
                {"model": OmegaConf.to_container(cfg["model_cfg"], resolve=True)}
            ),
            **flatten_mapping(
                {"trainer": OmegaConf.to_container(cfg["trainer_cfg"], resolve=True)}
            ),
        }
        pl_logger.log_hyperparams(params)

    logger.remove()
    logger.add(
        sys.stdout,
        format="<level>{level} {message}</level>",
        level="INFO",
        colorize=True,
    )
    log_file = Path(os.getcwd()) / "train.log"
    logger.add(log_file, level="INFO")
    logger.info(f"Log file at {log_file}")

    meta_data = {}
    meta_data["torch_version"] = str(torch.__version__)
    meta_data["date"] = str(datetime.now())
    meta_data["git"] = log_git(nndet.__path__[0], repo_name="nndet")
    meta_data["overwrites"] = str(ov)
    save_json(meta_data, "./meta.json")
    # try:
    #     write_requirements_to_file("requirements.txt")
    # except Exception as e:
    #     logger.error(f"Could not log req: {e}")

    plan_path = Path(str(cfg.host["plan_path"]))
    plan = load_pickle(plan_path)

    data_dir = (
        Path(cfg.host["preprocessed_output_dir"]) / plan["data_identifier"] / "imagesTr"
    )

    datamodule = Datamodule(
        io_cfg=OmegaConf.to_container(cfg["io_cfg"], resolve=True),
        augment_cfg=OmegaConf.to_container(cfg["augment_cfg"], resolve=True),
        plan=plan,
        data_dir=data_dir,
        fold=cfg["exp"]["fold"],
        log_aug=log_aug,
    )
    # copy IO config overwrites to plan
    plan["patch_size"] = list(datamodule.patch_size)
    plan["batch_size"] = int(datamodule.batch_size)

    module = MODULE_REGISTRY[cfg["module"]](
        model_cfg=OmegaConf.to_container(cfg["model_cfg"], resolve=True),
        trainer_cfg=OmegaConf.to_container(cfg["trainer_cfg"], resolve=True),
        plan=plan,
    )
    callbacks = []
    checkpoint_cb = ModelCheckpoint(
        dirpath=train_dir,
        filename="model_best",
        save_last=True,
        save_top_k=cfg["trainer_cfg"].get("save_top_k", 1),
        monitor=cfg["trainer_cfg"]["monitor_key"],
        mode=cfg["trainer_cfg"]["monitor_mode"],
    )
    checkpoint_cb.CHECKPOINT_NAME_LAST = "model_last"
    callbacks.append(checkpoint_cb)
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    OmegaConf.save(cfg, str(Path(os.getcwd()) / "config.yaml"))
    OmegaConf.save(cfg, str(Path(os.getcwd()) / "config_resolved.yaml"), resolve=True)
    save_pickle(plan, train_dir / "plan.pkl")  # backup plan
    save_json(create_debug_plan(plan), "./plan_debug.json")  # easy read backup
    splits = load_pickle(
        Path(cfg.host.preprocessed_output_dir) / datamodule.splits_file
    )
    save_pickle(splits, train_dir / "splits.pkl")

    trainer_kwargs = {}
    if cfg["exec"]["mode"].lower() == "resume":
        logger.info("Found train mode: resume -> will load checkpoint")
        trainer_kwargs["resume_from_checkpoint"] = train_dir / "model_last.ckpt"
    elif cfg["exec"]["mode"].lower() == "transfer":
        logger.info("Found train mode: transfer -> loading model weights")
        module.load_state_dict(
            torch.load(train_dir / "model_last.ckpt")["state_dict"], strict=True
        )

    num_gpus = cfg["trainer_cfg"]["gpus"]
    logger.info(f"Using {num_gpus} GPUs for training")

    plugins = []
    if p := cfg["trainer_cfg"].get("plugins", None):
        plugins.append(p)
    logger.info(f"Using {plugins} plugins for training")

    callbacks.append(ModelSummary(max_depth=10, log_net=log_net))

    if "terminate_on_nan" in cfg["trainer_cfg"]:
        detect_anomaly = cfg["trainer_cfg"]["terminate_on_nan"]
    elif "detect_anomaly" in cfg["trainer_cfg"]:
        detect_anomaly = cfg["trainer_cfg"]["detect_anomaly"]
    else:
        detect_anomaly = False

    if (
        cfg["trainer_cfg"]["precision"] == 16
        and cfg["trainer_cfg"]["amp_backend"] == "native"
    ):
        device = "cuda" if num_gpus > 0 else "cpu"
        precision_plugin = ExposedNativeMixedPrecisionPlugin(
            precision=16,
            device=device,
            init_scale=8192.0,
        )
        plugins.append(precision_plugin)

    trainer = pl.Trainer(
        gpus=list(range(num_gpus)) if num_gpus > 1 else num_gpus,
        accelerator=cfg["trainer_cfg"]["accelerator"],
        precision=cfg["trainer_cfg"]["precision"],
        amp_backend=cfg["trainer_cfg"]["amp_backend"],
        amp_level=cfg["trainer_cfg"]["amp_level"],
        benchmark=cfg["trainer_cfg"]["benchmark"],
        deterministic=cfg["trainer_cfg"]["deterministic"],
        callbacks=callbacks,
        logger=pl_logger,
        max_epochs=module.max_epochs,
        progress_bar_refresh_rate=None if bool(int(os.getenv("det_verbose", 1))) else 0,
        reload_dataloaders_every_epoch=False,
        num_sanity_val_steps=10,
        plugins=plugins,
        detect_anomaly=detect_anomaly,
        move_metrics_to_cpu=False,
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
        case_ids = splits[cfg["exp"]["fold"]]["val"]
        if (
            "debug" in cfg["trainer_cfg"]
            and "num_cases_val" in cfg["trainer_cfg"]["debug"]
        ):
            logger.warning(
                "[!!!] Detected debug mode for sweep using reduced set of cases"
            )
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
            target_dir = (
                train_dir / "val_predictions"
                if restore
                else train_dir / "val_predictions_preprocessed"
            )
            extract_results(
                source_dir=train_dir / "sweep_predictions",
                target_dir=target_dir,
                ensembler_cls=ensembler_cls,
                restore=restore,
                **inference_plan,
            )
        _evaluate(
            task=cfg["task"],
            model=cfg["exp"]["id"],
            fold=cfg["exp"]["fold"],
            test=False,
            do_case_eval=(
                module.requires_case_eval and (cfg["data"]["target_class"] is not None)
            ),
            do_boxes_eval=module.requires_box_eval(),
            do_analyze_boxes=module.requires_box_eval(),
            do_masks_eval=module.requires_mask_eval(),
            do_analyze_masks=module.requires_mask_eval(),
            do_seg_eval=module.requires_seg_eval(),
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

    logger.remove()
    logger.add(sys.stdout, format="{level} {message}", level="INFO")
    log_file = Path(os.getcwd()) / "sweep.log"
    logger.add(log_file, level="INFO")
    logger.info(f"Log file at {log_file}")

    plan = load_pickle(train_dir / "plan.pkl")
    data_dir = (
        Path(cfg.host["preprocessed_output_dir"]) / plan["data_identifier"] / "imagesTr"
    )

    module = MODULE_REGISTRY[cfg["module"]](
        model_cfg=OmegaConf.to_container(cfg["model_cfg"], resolve=True),
        trainer_cfg=OmegaConf.to_container(cfg["trainer_cfg"], resolve=True),
        plan=plan,
    )

    splits = load_pickle(train_dir / "splits.pkl")
    case_ids = splits[cfg["exp"]["fold"]]["val"]

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
        target_dir = (
            train_dir / "val_predictions"
            if restore
            else train_dir / "val_predictions_preprocessed"
        )
        extract_results(
            source_dir=train_dir / "sweep_predictions",
            target_dir=target_dir,
            ensembler_cls=ensembler_cls,
            restore=restore,
            **inference_plan,
        )

    _evaluate(
        task=cfg["task"],
        model=cfg["exp"]["id"],
        fold=cfg["exp"]["fold"],
        test=False,
        do_case_eval=(
            module.requires_case_eval and (cfg["data"]["target_class"] is not None)
        ),
        do_boxes_eval=module.requires_box_eval(),
        do_analyze_boxes=module.requires_box_eval(),
        do_masks_eval=module.requires_mask_eval(),
        do_analyze_masks=module.requires_mask_eval(),
        do_seg_eval=module.requires_seg_eval(),
    )


def _evaluate(
    task: str,
    model: str,
    fold: int,
    test: bool = False,
    do_case_eval: bool = False,
    do_boxes_eval: bool = False,
    do_masks_eval: bool = False,
    do_seg_eval: bool = False,
    do_analyze_boxes: bool = False,
    do_analyze_masks: bool = False,
    eval_preprocessed: bool = False,
):
    """
    This entrypoint runs the evaluation

    Args:
        task: current task
        model: full name of the model run determine empricial parameters for
            e.g. RetinaUNetV001_D3V001_3d
        fold: current fold
        test: use test split
        do_case_eval: evaluate patient metrics
        do_boxes_eval: perform box evaluation
        do_masks_eval: perform instance segmentation evaluation
        do_seg_eval: perform semantic segmentation evaluation
        do_analyze_boxes: run analysis of box results
        do_analyze_masks: run analysis of mask results
    """
    # prepare paths
    task = get_task(task, name=True)
    model_dir = Path(os.getenv("det_models")) / task / model
    training_dir = get_training_dir(model_dir, fold)

    data_dir_task = Path(os.getenv("det_data")) / task
    data_cfg = load_dataset_info(data_dir_task)

    prefix = "test" if test else "val"

    modes = [True]
    if not test and eval_preprocessed:
        modes.append(False)
    if test and eval_preprocessed:
        logger.warning(
            "Evaluation of preprocessed data only supported on validation data."
        )

    for restore in modes:
        if restore:
            pred_dir_name = f"{prefix}_predictions"
            gt_dir_name = "labelsTs" if test else "labelsTr"
            gt_dir = data_dir_task / "preprocessed" / gt_dir_name
        else:
            plan = load_pickle(training_dir / "plan.pkl")
            pred_dir_name = f"{prefix}_predictions_preprocessed"
            gt_dir = (
                data_dir_task / "preprocessed" / plan["data_identifier"] / "labelsTr"
            )

        pred_dir = training_dir / pred_dir_name
        save_dir = (
            training_dir / f"{prefix}_results"
            if restore
            else training_dir / f"{prefix}_results_preprocessed"
        )

        # compute metrics
        if do_boxes_eval:
            logger.info(f"Computing box metrics: restore {restore}")
            scores, curves = evaluate_box_dir(
                pred_dir=pred_dir,
                gt_dir=gt_dir,
                classes=list(data_cfg["labels"].keys()),
                save_dir=save_dir / "boxes",
            )
            save_metric_output(scores, curves, save_dir, "results_boxes")

        if do_case_eval:
            logger.info(f"Computing case metrics: restore {restore}")
            scores, curves = evaluate_case_dir(
                pred_dir=pred_dir,
                gt_dir=gt_dir,
                classes=list(data_cfg["labels"].keys()),
                target_class=data_cfg["target_class"],
            )
            save_metric_output(scores, curves, save_dir, "results_case")

        if do_seg_eval:
            raise NotImplementedError()
            logger.info(f"Computing seg metrics: restore {restore}")
            scores, curves = evaluate_seg_dir(
                pred_dir=pred_dir,
                gt_dir=gt_dir,
            )
            save_metric_output(scores, curves, save_dir, "results_seg")

        if do_masks_eval:
            logger.info(f"Computing mask metrics: restore {restore}")
            scores, curves = evaluate_mask_dir(
                pred_dir=pred_dir,
                gt_dir=gt_dir,
                classes=list(data_cfg["labels"].keys()),
                save_dir=save_dir / "masks",
            )
            save_metric_output(scores, curves, save_dir, "results_masks")

        # run analysis
        save_dir = (
            training_dir / f"{prefix}_analysis"
            if restore
            else training_dir / f"{prefix}_analysis_preprocessed"
        )
        if do_analyze_boxes:
            logger.info(f"Analyze box predictions: restore {restore}")
            run_analysis_suite(
                prediction_dir=pred_dir,
                gt_dir=gt_dir,
                save_dir=save_dir / "boxes",
            )
        if do_analyze_masks:
            logger.info("Analyze mask predictions is not implemented yet.")


if __name__ == "__main__":
    train()
