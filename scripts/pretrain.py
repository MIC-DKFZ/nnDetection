import argparse
import os
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch
from hydra.experimental import initialize_config_module
from loguru import logger
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

import nndet
from nndet.io.datamodule.module import PtDatamodule as Datamodule
from nndet.io.load import load_pickle, save_json, save_pickle
from nndet.io.paths import get_task
from nndet.ptmodule import MODULE_REGISTRY
from nndet.utils.check import env_guard
from nndet.utils.config import compose
from nndet.utils.info import (
    ModelSummary,
    create_debug_plan,
    flatten_mapping,
    log_git,
    write_requirements_to_file,
)


def init_train_dir(
    cfg,
    task: str,
    id: str,
    fold: int,
) -> Path:
    """
    Initialize training directory and make it the current working directory

    Args:
        task: task name
        id: experiment identifier
        fold: fold
    """
    # determine folder for experiment
    output_dir = Path(cfg.host.parent_results) / str(task) / str(id) / "transfer"

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


@env_guard
def pretrain():
    """
    Training entry
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pretask",
        type=str,
        help="Pretrain Task id e.g. Task12_LIDC OR 12 OR LIDC",
    )
    parser.add_argument(
        "targettask",
        type=str,
        help="Target Task id e.g. Task12_LIDC OR 12 OR LIDC",
    )
    parser.add_argument(
        "-o",
        "--overwrites",
        type=str,
        nargs="+",
        help="overwrites for config file",
        required=False,
    )

    args = parser.parse_args()
    pretask = args.pretask
    targettask = args.targettask
    ov = args.overwrites

    _pretrain(
        pretask=pretask,
        targettask=targettask,
        ov=ov,
    )


def _pretrain(
    pretask: str,
    targettask: str,
    ov: List[str],
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
    cfg = compose(pretask, "config.yaml", overrides=ov if ov is not None else [])

    pretask = get_task(pretask, name=True)
    targettask = get_task(targettask, name=True)

    assert cfg.host.parent_data is not None, "Parent data can not be None"
    assert cfg.host.parent_results is not None, "Output dir can not be None"

    train_dir = init_train_dir(cfg, targettask, cfg["exp"]["id"], cfg["exp"]["fold"])

    pl_logger = MLFlowLogger(
        experiment_name=cfg["task"],
        tags={
            "host": socket.gethostname(),
            "fold": cfg["exp"]["fold"],
            "task": cfg["task"],
            "job_id": os.getenv("LSB_JOBID", "no_id"),
            "mlflow.runName": cfg["exp"]["id"],
        },
        save_dir=os.getenv("MLFLOW_TRACKING_URI", "./mlruns"),
    )
    pl_logger.log_hyperparams(
        flatten_mapping(
            {"model": OmegaConf.to_container(cfg["model_cfg"], resolve=True)}
        )
    )
    pl_logger.log_hyperparams(
        flatten_mapping(
            {"trainer": OmegaConf.to_container(cfg["trainer_cfg"], resolve=True)}
        )
    )

    logger.remove()
    logger.add(sys.stdout, format="{level} {message}", level="INFO")
    log_file = Path(os.getcwd()) / "pretrain.log"
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

    det_data_path = Path(str(os.getenv("det_data")))
    pre_plan_path = det_data_path / pretask / "preprocessed" / f"{cfg['plan']}.pkl"
    target_plan_path = (
        det_data_path / targettask / "preprocessed" / f"{cfg['plan']}.pkl"
    )

    pre_plan = load_pickle(pre_plan_path)
    target_plan = load_pickle(target_plan_path)
    if pre_plan["num_classes"] != target_plan["num_classes"]:
        raise NotImplementedError("Not supported yet.")
    if pre_plan["num_modalities"] > target_plan["num_modalities"]:
        raise NotImplementedError("Not supported yet.")
    # update properties from target plan
    # use same patch size to make sure that network config will work
    pre_plan["patch_size"] = target_plan["patch_size"]
    # pre_plan["batch_size"] = target_plan["batch_size"]
    pre_plan["architecture"] = target_plan["architecture"]
    pre_plan["anchors"] = target_plan["anchors"]
    save_json(create_debug_plan(pre_plan), "./plan_debug.json")

    data_dir = (
        Path(cfg.host["preprocessed_output_dir"])
        / pre_plan["data_identifier"]
        / "imagesTr"
    )

    datamodule = Datamodule(
        io_cfg=OmegaConf.to_container(cfg["io_cfg"], resolve=True),
        augment_cfg=OmegaConf.to_container(cfg["augment_cfg"], resolve=True),
        plan=pre_plan,
        data_dir=data_dir,
        fold=cfg["exp"]["fold"],
    )
    module = MODULE_REGISTRY[cfg["module"]](
        model_cfg=OmegaConf.to_container(cfg["model_cfg"], resolve=True),
        trainer_cfg=OmegaConf.to_container(cfg["trainer_cfg"], resolve=True),
        plan=pre_plan,
    )
    callbacks = []
    checkpoint_cb = ModelCheckpoint(
        dirpath=train_dir,
        filename="model_best",
        save_last=True,
        save_top_k=1,
        monitor=cfg["trainer_cfg"]["monitor_key"],
        mode=cfg["trainer_cfg"]["monitor_mode"],
    )
    checkpoint_cb.CHECKPOINT_NAME_LAST = "model_last"
    callbacks.append(checkpoint_cb)
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    callbacks.append(ModelSummary(max_depth=10))

    # save configs
    OmegaConf.save(cfg, str(Path(os.getcwd()) / "pre_config.yaml"))
    OmegaConf.save(
        cfg, str(Path(os.getcwd()) / "pre_config_resolved.yaml"), resolve=True
    )

    cfg_target = compose(
        targettask, "config.yaml", overrides=ov if ov is not None else []
    )
    OmegaConf.save(cfg_target, str(Path(os.getcwd()) / "config.yaml"))
    OmegaConf.save(
        cfg_target, str(Path(os.getcwd()) / "config_resolved.yaml"), resolve=True
    )

    # save plans
    save_pickle(target_plan, train_dir / "plan.pkl")  # save plan for downstream task
    save_pickle(pre_plan, train_dir / "pre_plan.pkl")  # backup plan

    splits = load_pickle(
        Path(cfg.host.preprocessed_output_dir) / datamodule.splits_file
    )
    save_pickle(splits, train_dir / "pre_splits.pkl")

    trainer_kwargs = {}
    if cfg["exec"]["mode"].lower() == "resume":
        raise NotImplementedError(
            "Resume training not implemented for pretask training."
        )
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
        num_sanity_val_steps=10,  # 10,
        plugins=plugins,
        detect_anomaly=True,
        move_metrics_to_cpu=True,
        **trainer_kwargs,
    )
    trainer.fit(module, datamodule=datamodule)
