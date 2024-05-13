# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pytorch_lightning as pl
import torch
from loguru import logger

from nndet.core.abstract import AbstractDetector
from nndet.io.transforms import Compose
from nndet.ptmodule.optimizer import OPTIMIZER_REGISTRY
from nndet.training.callbacks import CheckWeightsNaN, EpochTimerCallback, LossNaNError
from nndet.training.swa import SWACycleLinear
from nndet.utils.check import check_torch_version


class LightningBaseModule(pl.LightningModule):
    def __init__(
        self,
        model_cfg: dict,
        trainer_cfg: dict,
        plan: dict,
        accelerator_cfg: Optional[dict] = None,
        **kwargs,
    ):
        """
        Provides a base module which is used inside of nnDetection.
        All lightning modules of nnDetection should be derifed from this!

        Args:
            model_cfg: model configuration. Check ::method::`from_config_plan`
                for more information
            trainer_cfg: trainer information
            plan: contains parameters which were derived from the planning
                stage
            accelerator_cfg: optionally provide additional information
                on accelerator configuration, e.g. if module should be
                compiled
        """
        super().__init__(**kwargs)
        self.model_cfg = model_cfg
        self.trainer_cfg = trainer_cfg
        self.accelerator_cfg = accelerator_cfg
        self.plan = plan
        self.dim = len(plan["patch_size"])
        assert self.dim in [2, 3]
        self.do_channels_last = False

        # initialize model
        self.model: AbstractDetector = self.from_config_plan(
            model_cfg=self.model_cfg,
            plan_arch=self.plan["architecture"],
            plan_anchors=self.plan["anchors"],
            patch_size=plan["patch_size"],
        )
        self._init_accelerator_cfg()

        # prepare tansformations for input
        self.pre_trafo = self._init_trafo()
        logger.info(f"Lightningmodule running pre transforms \n: {self.pre_trafo}")

        # initialize evaluation
        self.evaluators = self.evaluation_init(plan=plan)
        _tmp = {key: item.__class__.__name__ for key, item in self.evaluators.items()}
        logger.info(f"Lightningmodule running evaluators: {_tmp}")

        # define key for sweeping
        self.sweep_key = self.trainer_cfg["sweep_key"]
        self.monitor_key = self.trainer_cfg["monitor_key"]
        logger.info(f"Using {self.sweep_key} for sweeping and {self.monitor_key} for monitoring.")

        # setup other variables
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.example_input_array_shape = (
            1,
            plan["architecture"]["in_channels"],
            *plan["patch_size"],
        )

    def _init_accelerator_cfg(self) -> None:
        """
        Apply additional actions to configure module for additional speedups
        e.g. channels_last(_3d) memory format or torch.compile
        """
        if self.accelerator_cfg is None:
            self.do_channels_last = False
            return

        # channels last memory format
        if self.accelerator_cfg.get("do_channels_last", False):
            self.do_channels_last = self.accelerator_cfg["do_channels_last"]
            if self.do_channels_last:
                logger.info("PtModule uses channels_last memory format")
                if self.dim == 3:
                    self.model = self.model.to(memory_format=torch.channels_last_3d)
                else:
                    self.model = self.model.to(memory_format=torch.channels_last)

        # compile model
        if self.accelerator_cfg.get("do_compile", False):
            if check_torch_version(major_version=2):
                _kwargs = self.accelerator_cfg.get("compile", {})
                logger.info(f"PtModule uses torch.compile with arguments: {_kwargs}")
                self.model = torch.compile(self.model, **_kwargs)
            else:
                logger.error(
                    "Torch compile was enabled in config but minimal "
                    "PyTorch Version of 2.0.0 was not met! Skipping compile."
                )

    def _init_trafo(self) -> Callable:
        """
        Initialize pre transforms from Mixin
        """
        trafos = self.get_pre_transforms(plan=self.plan)
        return Compose(trafos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used to generate summary
        Do not(!) use this for inference. This will only forward
        the input through the network which does not include
        detection spcific postprocessing!
        """
        return self.model.inference_step(x)
        # return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Computes a single training step
        See :class:`BaseRetinaNet` for more information
        """
        with torch.no_grad():
            batch = self.pre_trafo(**batch)

        if "target" in batch:  # free memory from numbered instance seg
            del batch["target"]
        targets = {key: item for key, item in batch.items() if "target_" in key}
        if "target_seg" in targets:
            # [optional] add semantic segmentation to targets if available
            # Remove channel dimension of semantic segmentation
            targets["target_seg"] = targets["target_seg"][:, 0]
        if "target_binary_masks" in targets:
            # [optional] add bianry masks to targets if available
            targets["target_binary_masks"] = targets["target_binary_masks"]

        if self.do_channels_last:
            if self.dim == 3:
                _data = batch["data"].to(memory_format=torch.channels_last_3d)
            else:
                _data = batch["data"].to(memory_format=torch.channels_last)
        else:
            _data = batch["data"]

        losses = self.model.train_step(
            images=_data,
            targets=targets,
            batch_num=batch_idx,
        )

        # Exclude logging keys starting with __
        info = {key: losses.pop(key) for key in list(losses.keys()) if key.startswith("__")}
        loss = sum(losses.values())

        if torch.isnan(loss):
            raise LossNaNError("Found NaN loss in training step.")

        out = {"loss": loss.detach().item(), **{f"loss_{key}": l.detach().item() for key, l in losses.items()}, **info}
        self.log("train_step_loss", out["loss"], prog_bar=True, logger=False, batch_size=1)
        self.training_step_outputs.append(out)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Computes a single validation step (same as train step but with
        additional prediciton processing)
        See ::class::`BaseRetinaNet` for more information
        """
        with torch.no_grad():
            batch = self.pre_trafo(**batch)

            if "target" in batch:  # free memory from numbered instance seg
                del batch["target"]
            targets = {key: item for key, item in batch.items() if "target_" in key}
            if "target_seg" in targets:
                # [optional] add semantic segmentation to targets if available
                # Remove channel dimension of semantic segmentation
                targets["target_seg"] = targets["target_seg"][:, 0]
            if "target_binary_masks" in targets:
                # [optional] add bianry masks to targets if available
                targets["target_binary_masks"] = targets["target_binary_masks"]

            if self.do_channels_last:
                if self.dim == 3:
                    _data = batch["data"].to(memory_format=torch.channels_last_3d)
                else:
                    _data = batch["data"].to(memory_format=torch.channels_last)
            else:
                _data = batch["data"]

            losses, predictions = self.model.validation_step(
                images=_data,
                targets=targets,
                batch_num=batch_idx,
            )
            # Exclude criterion logging keys starting with __
            info = {key: losses.pop(key) for key in list(losses.keys()) if key.startswith("__")}
            loss = sum(losses.values())

        super().evaluation_step(predictions=predictions, targets=targets)

        out = {
            "loss": loss.detach().item(),
            **{f"loss_{key}": l.detach().item() for key, l in losses.items()},
            **info,
        }
        self.log("val_step_loss", out["loss"], prog_bar=True, logger=False, batch_size=1)
        self.validation_step_outputs.append(out)
        return loss

    def on_train_epoch_end(self):
        """
        Log train loss to loguru logger
        """
        # process and log losses
        vals = defaultdict(list)
        for _val in self.training_step_outputs:
            for _k, _v in _val.items():
                vals[_k].append(_v)

        _log_loss_str = "Train:"
        for _key, _vals in vals.items():
            mean_val = sum(_vals) / len(_vals)
            if _key.startswith("loss"):
                _log_loss_str = _log_loss_str + f" {_key} {mean_val:0.5f}"

            if _key.startswith("__"):
                self.log(f"train_info/{_key}", mean_val, sync_dist=True, prog_bar=False, logger=True, batch_size=1)
            else:
                self.log(f"train_loss/{_key}", mean_val, sync_dist=True, prog_bar=False, logger=True, batch_size=1)
        logger.info(_log_loss_str)

        self.training_step_outputs.clear()  # free memory
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self):
        """
        Log val loss to loguru logger
        """
        # process and log losses
        vals = defaultdict(list)
        for _val in self.validation_step_outputs:
            for _k, _v in _val.items():
                vals[_k].append(_v)

        _log_loss_str = "Val:"
        for _key, _vals in vals.items():
            mean_val = sum(_vals) / len(_vals)
            if _key.startswith("loss"):
                _log_loss_str = _log_loss_str + f" {_key} {mean_val:0.5f}"

            if _key.startswith("__"):
                self.log(f"val_info/{_key}", mean_val, sync_dist=True, prog_bar=False, logger=True, batch_size=1)
            else:
                self.log(f"val_loss/{_key}", mean_val, sync_dist=True, prog_bar=False, logger=True, batch_size=1)
        logger.info(_log_loss_str)

        # process and log metrics
        super().evaluation_end()

        self.validation_step_outputs.clear()  # free memory
        return super().on_validation_epoch_end()

    @property
    def train_epochs(self):
        """
        Return number of train epochs
        """
        if "max_num_epochs" in self.plan:
            epochs = self.plan["max_num_epochs"]
            logger.info(f"Using max epochs {epochs} from plan.")
        else:
            epochs = self.trainer_cfg["max_num_epochs"]
            logger.info(f"Using max epochs {epochs} from config.")
        return epochs

    @property
    def max_epochs(self):
        """
        Number of epochs of full training
        """
        return self.train_epochs + self.trainer_cfg.get("swa_epochs", 0)

    @property
    def example_input_array(self):
        """
        Create example input
        """
        return torch.zeros(*self.example_input_array_shape)

    def inference_step(self, batch: Any, **kwargs) -> Dict[str, Any]:
        """
        Prediction method used by nnDetection predictor class
        """
        return self.model.inference_step(batch, **kwargs)

    def configure_optimizers(self):
        """
        Configure optimizer and scheduler
        """
        return OPTIMIZER_REGISTRY[self.trainer_cfg["opt_class"]].configure_optimizers(self)

    def configure_callbacks(self):
        """
        Configure default callbacks.
        Per default the epoch timer is added to measure training and
        validation time. Optionally a Stochastic Weight Averaging
        Callback can be added by configuring `swa_epoch` with a
        cyclic learning rate which oscilates between `initial_lr / 10`
        and `initial_lr / 1000` once per epoch.

        Configuration keys:
            ``"swa_epochs"`` int
                number of epoch to perform SWA. The model will be snapshotted
                at the end of each epoch.
            ``"initial_lr"`` float
                initial learning rate of optimizer
            ``"num_train_batches_per_epoch"`` int
                number of train batches per epoch.
        """
        callbacks = super().configure_callbacks()
        callbacks.append(EpochTimerCallback())
        callbacks.append(CheckWeightsNaN())

        if e := self.trainer_cfg.get("swa_epochs", 0) > 0:
            logger.info(f"Training with SWA, found {e} swa epochs.")
            callbacks.append(
                SWACycleLinear(
                    swa_epoch_start=self.train_epochs,
                    cycle_initial_lr=self.trainer_cfg["initial_lr"] / 10.0,
                    cycle_final_lr=self.trainer_cfg["initial_lr"] / 1000.0,
                    num_iterations_per_epoch=self.trainer_cfg["num_train_batches_per_epoch"],
                )
            )
        return callbacks

    def load_custom_state_dict(self, path: os.Pathlike) -> None:
        """
        Load custom state_dict

        Args:
            path: filepath to model checkpoint
        """

        path = Path(path)
        if not path.is_file():
            _s = f"Path {path} for checkpoint for transfer learning does not exist."
            logger.error(_s)
            raise RuntimeError(_s)

        checkpoint = torch.load(str(path), map_location="cpu")
        self.load_state_dict(checkpoint["state_dict"], strict=True)
        return
