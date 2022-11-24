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
import sys
import socket
import argparse
import importlib
from pathlib import Path
from datetime import datetime
from typing import List

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from loguru import logger
from hydra import initialize_config_module
from omegaconf.omegaconf import OmegaConf

import nndet
from nndet.utils.config import compose, load_dataset_info
from nndet.utils.info import log_git, write_requirements_to_file, \
    create_debug_plan, flatten_mapping
from nndet.utils.check import env_guard
from nndet.utils.analysis import run_analysis_suite
from nndet.io.datamodule.bg_module import Datamodule
from nndet.io.paths import get_task, get_training_dir
from nndet.io.load import load_pickle, save_json, save_pickle
from nndet.evaluator.registry import save_metric_output, evaluate_box_dir, \
    evaluate_case_dir, evaluate_seg_dir
from nndet.inference.ensembler.base import extract_results
from nndet.ptmodule import MODULE_REGISTRY


@env_guard
def train():
    """
    Training entry
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str,
                        help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument('-o', '--overwrites', type=str, nargs='+',
                        help="overwrites for config file",
                        required=False)
    parser.add_argument('--sweep',
                        help="Run empirical parameter optimization",
                        action='store_true',
                        )

    args = parser.parse_args()
    task = args.task
    ov = args.overwrites
    do_sweep = args.sweep
    _train(
        task=task,
        ov=ov,
        do_sweep=do_sweep,
        )


@env_guard
def sweep():
    """
    Sweep entry
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str,
                        help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument('model', type=str,
                        help="full name of experiment to sweep e.g. RetinaUNetV0_D3V001_3d")
    parser.add_argument('fold', type=int,
                        help="experiment fold")
    args = parser.parse_args()
    task = args.task
    model = args.model
    fold = args.fold
    _sweep(
        task=task,
        model=model,
        fold=fold,
        )


@env_guard
def evaluate(): 
    """
    Evaluation entry

    seg, instances are not supported yet
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument('model', type=str, help="model name, e.g. RetinaUNetV0_D3V001_3d")
    parser.add_argument('fold', type=int, help="fold, -1 => consolidated")

    parser.add_argument('--test',
                        help="Evaluate test predictions -> uses different folder",
                        action='store_true')
    parser.add_argument('--case', help="Run Case Evaluation", action='store_true')
    parser.add_argument('--boxes', help="Run Box Evaluation", action='store_true')
    parser.add_argument('--seg', help="Run Box Evaluation", action='store_true')
    parser.add_argument('--instances', help="Run Box Evaluation", action='store_true')
    parser.add_argument('--analyze_boxes', help="Run Box Evaluation", action='store_true')

    args = parser.parse_args()
    model = args.model
    fold = args.fold
    task = args.task
    test = args.test

    do_boxes_eval = args.boxes    
    do_case_eval = args.case
    do_seg_eval = args.seg
    do_instances_eval = args.instances

    do_analyze_boxes = args.analyze_boxes
    
    _evaluate(
        task=task,
        model=model,
        fold=fold,
        test=test,
        do_boxes_eval=do_boxes_eval,
        do_case_eval=do_case_eval,
        do_seg_eval=do_seg_eval,
        do_instances_eval=do_instances_eval,
        do_analyze_boxes=do_analyze_boxes,
    )


def init_train_dir(cfg) -> Path:
    """
    Initialize training directory and make it the current working directory
    """
    # determine folder for experiment
    output_dir = Path(cfg.host.parent_results) / str(cfg.task) / str(cfg.exp.id) / f"fold{cfg.exp.fold}"

    if cfg["train"]["mode"].lower() == "overwrite":
        if output_dir.is_dir():
            print(f"Found existing folder {output_dir}, this run will overwrite "
                  f"the results inside that folder")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        if not output_dir.is_dir():
            raise ValueError(f"{output_dir} is not a valid training dir and thus can not be resumed")
    os.chdir(str(output_dir))
    return output_dir


def _train(
    task: str,
    ov: List[str],
    do_sweep: bool,
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

    assert cfg.host.parent_data is not None, 'Parent data can not be None'
    assert cfg.host.parent_results is not None, 'Output dir can not be None'

    train_dir = init_train_dir(cfg)

    pl_logger = MLFlowLogger(
        experiment_name=cfg["task"],
        tags={
            "host": socket.gethostname(),
            "fold": cfg["exp"]["fold"],
            "task": cfg["task"],
            "job_id": os.getenv('LSB_JOBID', 'no_id'),
            "mlflow.runName": cfg["exp"]["id"],
            },
        save_dir=os.getenv("MLFLOW_TRACKING_URI", "./mlruns"),
    )
    pl_logger.log_hyperparams(flatten_mapping(
        {"model": OmegaConf.to_container(cfg["model_cfg"], resolve=True)}))
    pl_logger.log_hyperparams(flatten_mapping(
        {"trainer": OmegaConf.to_container(cfg["trainer_cfg"], resolve=True)}))

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
    save_json(meta_data, "./meta.json")
    try:
        write_requirements_to_file("requirements.txt")
    except Exception as e:
        logger.error(f"Could not log req: {e}")

    plan_path = Path(str(cfg.host["plan_path"]))
    plan = load_pickle(plan_path)
    save_json(create_debug_plan(plan), "./plan_debug.json")

    data_dir = Path(cfg.host["preprocessed_output_dir"]) / plan["data_identifier"] / "imagesTr"

    datamodule = Datamodule(
            augment_cfg=OmegaConf.to_container(cfg["augment_cfg"], resolve=True),
            plan=plan,
            data_dir=data_dir,
            fold=cfg["exp"]["fold"],
        )
    module = MODULE_REGISTRY[cfg["module"]](
        model_cfg=OmegaConf.to_container(cfg["model_cfg"], resolve=True),
        trainer_cfg=OmegaConf.to_container(cfg["trainer_cfg"], resolve=True),
        plan=plan,
        )
    callbacks = []
    checkpoint_cb = ModelCheckpoint(
        dirpath=train_dir,
        filename='model_best',
        save_last=True,
        save_top_k=1,
        monitor=cfg["trainer_cfg"]["monitor_key"],
        mode=cfg["trainer_cfg"]["monitor_mode"],
    )
    checkpoint_cb.CHECKPOINT_NAME_LAST = 'model_last'
    callbacks.append(checkpoint_cb)
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    OmegaConf.save(cfg, str(Path(os.getcwd()) / "config.yaml"))
    OmegaConf.save(cfg, str(Path(os.getcwd()) / "config_resolved.yaml"), resolve=True)
    save_pickle(plan, train_dir / "plan.pkl") # backup plan
    splits = load_pickle(Path(cfg.host.preprocessed_output_dir) / datamodule.splits_file)
    save_pickle(splits, train_dir / "splits.pkl")

    trainer_kwargs = {}
    if cfg["train"]["mode"].lower() == "resume":
        trainer_kwargs["resume_from_checkpoint"] = train_dir / "model_last.ckpt"

    num_gpus = cfg["trainer_cfg"]["gpus"]
    logger.info(f"Using {num_gpus} GPUs for training")
    plugins = cfg["trainer_cfg"].get("plugins", None)
    logger.info(f"Using {plugins} plugins for training")

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
        weights_summary='full',
        plugins=plugins,
        terminate_on_nan=True,  # TODO: make modular
        move_metrics_to_cpu=False,
        **trainer_kwargs
    )
    trainer.fit(module, datamodule=datamodule)

    if do_sweep:
        case_ids = splits[cfg["exp"]["fold"]]["val"]
        if "debug" in cfg and "num_cases_val" in cfg["debug"]:
            case_ids = case_ids[:cfg["debug"]["num_cases_val"]]

        inference_plan = module.sweep(
            cfg=OmegaConf.to_container(cfg, resolve=True),
            save_dir=train_dir,
            train_data_dir=data_dir,
            case_ids=case_ids,
            run_prediction=True,
        )

        plan["inference_plan"] = inference_plan
        save_pickle(plan, train_dir / "plan_inference.pkl")

        ensembler_cls = module.get_ensembler_cls(
            key="boxes", dim=plan["network_dim"]) # TODO: make this configurable    
        for restore in [True, False]:
            target_dir = train_dir / "val_predictions" if restore else \
                train_dir / "val_predictions_preprocessed"
            extract_results(source_dir=train_dir / "sweep_predictions",
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
            do_boxes_eval=True, # TODO: make this configurable
            do_analyze_boxes=True, # TODO: make this configurable
        )


def _sweep(
    task: str,
    model: str,
    fold: int,
    ):
    """
    Determine best postprocessing parameters for a trained model

    Args:
        task: current task
        model: full name of the model run determine empricial parameters for
            e.g. RetinaUNetV001_D3V001_3d
        fold: current fold
    """
    nndet_data_dir = Path(os.getenv("det_models"))
    task = get_task(task, name=True, models=True)
    train_dir = nndet_data_dir / task / model / f"fold{fold}"

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
    data_dir = Path(cfg.host["preprocessed_output_dir"]) / plan["data_identifier"] / "imagesTr"

    module = MODULE_REGISTRY[cfg["module"]](
        model_cfg=OmegaConf.to_container(cfg["model_cfg"], resolve=True),
        trainer_cfg=OmegaConf.to_container(cfg["trainer_cfg"], resolve=True),
        plan=plan,
        )

    splits = load_pickle(train_dir / "splits.pkl")
    case_ids = splits[cfg["exp"]["fold"]]["val"]
    inference_plan = module.sweep(
        cfg=OmegaConf.to_container(cfg, resolve=True),
        save_dir=train_dir,
        train_data_dir=data_dir,
        case_ids=case_ids,
        run_prediction=True, # TODO: add commmand line arg
    )

    plan["inference_plan"] = inference_plan
    save_pickle(plan, train_dir / "plan_inference.pkl")

    ensembler_cls = module.get_ensembler_cls(
        key="boxes", dim=plan["network_dim"]) # TODO: make this configurable    
    for restore in [True, False]:
        target_dir = train_dir / "val_predictions" if restore else \
            train_dir / "val_predictions_preprocessed"
        extract_results(source_dir=train_dir / "sweep_predictions",
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
        do_boxes_eval=True, # TODO: make this configurable
        do_analyze_boxes=True, # TODO: make this configurable
    )


def _evaluate(
    task: str,
    model: str,
    fold: int,
    test: bool = False,
    do_case_eval: bool = False,
    do_boxes_eval: bool = False,
    do_seg_eval: bool = False,
    do_instances_eval: bool = False,
    do_analyze_boxes: bool = False,
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
        do_seg_eval: perform semantic segmentation evaluation
        do_instances_eval: perform instance segmentation evaluation
        do_analyze_boxes: run analysis of box results
    """
    # prepare paths
    task = get_task(task, name=True)
    model_dir = Path(os.getenv("det_models")) / task / model
    training_dir = get_training_dir(model_dir, fold)

    data_dir_task = Path(os.getenv("det_data")) / task
    data_cfg = load_dataset_info(data_dir_task)

    prefix = "test" if test else "val"

    modes = [True] if test else [True, False]
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
        save_dir = training_dir / f"{prefix}_results" if restore else \
            training_dir / f"{prefix}_results_preprocessed"

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
            logger.info(f"Computing seg metrics: restore {restore}")
            scores, curves = evaluate_seg_dir(
                pred_dir=pred_dir,
                gt_dir=gt_dir,
                )
            save_metric_output(scores, curves, save_dir, "results_seg")
        if do_instances_eval:
            raise NotImplementedError

        # run analysis
        save_dir = training_dir / f"{prefix}_analysis" if restore else \
            training_dir / f"{prefix}_analysis_preprocessed"
        if do_analyze_boxes:
            logger.info(f"Analyze box predictions: restore {restore}")
            run_analysis_suite(prediction_dir=pred_dir,
                               gt_dir=gt_dir,
                               save_dir=save_dir / "boxes",
                               )


if __name__ == "__main__":
    train()
