# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, TypeVar

import pytorch_lightning as pl
import torch
from loguru import logger

from nndet.core.abstract import AbstractDetector
from nndet.io.transforms import Compose, TransferInputChannel
from nndet.ptmodule.optimizer import OPTIMIZER_REGISTRY
from nndet.training.callbacks import CheckWeightsNaN, EpochTimerCallback
from nndet.training.swa import SWACycleLinear


class LightningBaseModule(pl.LightningModule):
    def __init__(
        self,
        model_cfg: dict,
        trainer_cfg: dict,
        plan: dict,
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
        """
        super().__init__(**kwargs)
        self.model_cfg = model_cfg
        self.trainer_cfg = trainer_cfg
        self.plan = plan

        # determine shape for network visualisation
        self.example_input_array_shape = (
            1,
            plan["architecture"]["in_channels"],
            *plan["patch_size"],
        )

        # initialize model
        self.model: AbstractDetector = self.from_config_plan(
            model_cfg=self.model_cfg,
            plan_arch=self.plan["architecture"],
            plan_anchors=self.plan["anchors"],
            patch_size=plan["patch_size"],
        )

        # initialize pre transforms from ModeMixin
        trafos = self.get_pre_transforms(plan=plan)

        # handle transfer learning
        data_channels = self.plan["num_modalities"]  # number of channels of source data
        network_channels = self.plan["architecture"][
            "in_channels"
        ]  # number of channels of target data
        if network_channels > data_channels:
            logger.info(
                "Detected Transfer Learning Setup with different soruce "
                "and target channels. Adding additional transformation."
            )
            trafos.append(
                TransferInputChannel(
                    out_channels=network_channels,
                    data_key="data",
                )
            )

        self.pre_trafo = Compose(trafos)
        logger.info(f"Lightningmodule running pre transforms \n: {self.pre_trafo}")

        # initialize evaluation
        self.evaluators = self.evaluation_init(plan=plan)
        _tmp = {key: item.__class__.__name__ for key, item in self.evaluators.items()}
        logger.info(f"Lightningmodule running evaluators: {_tmp}")

        # define key for sweeping
        self.sweep_key = self.trainer_cfg["sweep_key"]
        self.monitor_key = self.trainer_cfg["monitor_key"]
        logger.info(
            f"Using {self.sweep_key} for sweeping and {self.monitor_key} for monitoring."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used to generate summary
        Do not(!) use this for inference. This will only forward
        the input through the network which does not include
        detection spcific postprocessing!
        """
        return self.model.inference_step(x)  # FIXME
        # return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Computes a single training step
        See :class:`BaseRetinaNet` for more information
        """
        with torch.no_grad():
            batch = self.pre_trafo(**batch)

        targets = {key: item for key, item in batch.items() if "target_" in key}
        if "target_seg" in targets:
            # [optional] add semantic segmentation to targets if available
            # Remove channel dimension of semantic segmentation
            targets["target_seg"] = targets["target_seg"][:, 0]
        if "target_binary_masks" in targets:
            # [optional] add bianry masks to targets if available
            targets["target_binary_masks"] = targets["target_binary_masks"]

        losses = self.model.train_step(
            images=batch["data"],
            targets=targets,
            batch_num=batch_idx,
        )
        loss = sum(losses.values())

        # self.log_dict(losses, prog_bar=True)

        return {"loss": loss, **{key: l.detach().item() for key, l in losses.items()}}

    def validation_step(self, batch, batch_idx):
        """
        Computes a single validation step (same as train step but with
        additional prediciton processing)
        See ::class::`BaseRetinaNet` for more information
        """
        with torch.no_grad():
            batch = self.pre_trafo(**batch)

            targets = {key: item for key, item in batch.items() if "target_" in key}
            if "target_seg" in targets:
                # [optional] add semantic segmentation to targets if available
                # Remove channel dimension of semantic segmentation
                targets["target_seg"] = targets["target_seg"][:, 0]
            if "target_binary_masks" in targets:
                # [optional] add bianry masks to targets if available
                targets["target_binary_masks"] = targets["target_binary_masks"]

            losses, predictions = self.model.validation_step(
                images=batch["data"],
                targets=targets,
                batch_num=batch_idx,
            )
            loss = sum(losses.values())

        # self.log_dict(losses, prog_bar=True)

        super().evaluation_step(predictions=predictions, targets=targets)

        return {
            "loss": loss.detach().item(),
            **{key: l.detach().item() for key, l in losses.items()},
        }

    def training_epoch_end(self, training_step_outputs):
        """
        Log train loss to loguru logger
        """
        # process and log losses
        vals = defaultdict(list)
        for _val in training_step_outputs:
            for _k, _v in _val.items():
                if _k == "loss":
                    vals[_k].append(_v.detach().item())
                else:
                    vals[_k].append(_v)

        for _key, _vals in vals.items():
            mean_val = sum(_vals) / len(_vals)
            if _key == "loss":
                logger.info(f"Train loss reached: {mean_val:0.5f}")
            self.log(f"train_loss/{_key}", mean_val, sync_dist=True)
        return super().training_epoch_end(training_step_outputs)

    def validation_epoch_end(self, validation_step_outputs):
        """
        Log val loss to loguru logger
        """
        # process and log losses
        vals = defaultdict(list)
        for _val in validation_step_outputs:
            for _k, _v in _val.items():
                vals[_k].append(_v)

        for _key, _vals in vals.items():
            mean_val = sum(_vals) / len(_vals)
            if _key == "loss":
                logger.info(f"Val loss reached: {mean_val:0.5f}")
            self.log(f"val_loss/{_key}", mean_val, sync_dist=True)

        # process and log metrics
        super().evaluation_end()

        # metrics are logged by respective evaluator
        # for key, item in metric_scores.items():
        #     self.log(f"val/{key}", item, prog_bar=False, logger=True, sync_dist=True)

        return super().validation_epoch_end(validation_step_outputs)

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
        return OPTIMIZER_REGISTRY[self.trainer_cfg["opt_class"]].configure_optimizers(
            self
        )

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
                    num_iterations_per_epoch=self.trainer_cfg[
                        "num_train_batches_per_epoch"
                    ],
                )
            )
        return callbacks


LightningBaseModuleType = TypeVar("LightningBaseModuleType", bound=LightningBaseModule)
