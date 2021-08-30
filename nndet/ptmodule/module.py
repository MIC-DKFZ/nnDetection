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

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Optional, TypeVar

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.core.memory import ModelSummary

from nndet.io.load import save_txt
from nndet.io.transforms import Compose, TransferInputChannel
from nndet.training.misc import EpochTimerCallback
from nndet.training.swa import SWACycleLinear


class LightningBaseModule(pl.LightningModule):
    def __init__(self, model_cfg: dict, trainer_cfg: dict, plan: dict, **kwargs):
        """
        Provides a base module which is used inside of nnDetection.
        All lightning modules of nnDetection should be derifed from this!

        Args:
            model_cfg: model configuration. Check :method:`from_config_plan`
                for more information
            trainer_cfg: trainer information
            plan: contains parameters which were derived from the planning
                stage
        """
        super().__init__()
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
        self.model = self.from_config_plan(
            model_cfg=self.model_cfg,
            plan_arch=self.plan["architecture"],
            plan_anchors=self.plan["anchors"],
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
        logger.info(f"Lightningmodule running evaluators: {self.evaluators}")

        # define key for sweeping
        if self.trainer_cfg["monitor_key"].startswith("val/"):
            self.eval_score_key = str(self.trainer_cfg["monitor_key"]).split("/", 1)[1]
        else:
            self.eval_score_key = self.trainer_cfg
        logger.info(f"Using {self.eval_score_key} for sweeping.")

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

        losses, _ = self.model.train_step(  # FIXME
            images=batch["data"],
            targets={
                "target_boxes": batch["boxes"],
                "target_classes": batch["classes"],
                "target_seg": batch["target"][:, 0],  # Remove channel dimension
            },
            predict=False,
            batch_num=batch_idx,
        )
        loss = sum(losses.values())

        # self.log_dict(losses, prog_bar=True)

        return {"loss": loss, **{key: l.detach().item() for key, l in losses.items()}}

    def validation_step(self, batch, batch_idx):
        """
        Computes a single validation step (same as train step but with
        additional prediciton processing)
        See :class:`BaseRetinaNet` for more information
        """
        with torch.no_grad():
            batch = self.pre_trafo(**batch)
            targets = {
                "target_boxes": batch["boxes"],
                "target_classes": batch["classes"],
                "target_seg": batch["target"][:, 0],  # Remove channel dimension
            }
            losses, predictions = self.model.train_step(  # FIXME
                images=batch["data"],
                targets=targets,
                predict=True,
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
        metric_scores = super().evaluation_end()

        for key, item in metric_scores.items():
            self.log(f"val/{key}", item, prog_bar=False, logger=True, sync_dist=True)

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

    def summarize(self, *args, **kwargs) -> Optional[ModelSummary]:
        """
        Save model summary as txt
        """
        summary = super().summarize(*args, **kwargs)
        save_txt(summary, "./network")
        return summary

    def inference_step(self, batch: Any, **kwargs) -> Dict[str, Any]:
        """
        Prediction method used by nnDetection predictor class
        """
        return self.model.inference_step(batch, **kwargs)

    # @classmethod
    # def from_config_plan(  # FIXME
    #     cls,
    #     model_cfg: dict,
    #     plan_arch: dict,
    #     plan_anchors: dict,
    #     log_num_anchors: str = None,
    #     **kwargs,
    # ):
    #     """
    #     Used to generate the model
    #     """
    #     raise NotImplementedError

    # @staticmethod
    # def get_ensembler_cls(key: Hashable, dim: int) -> Callable: # TODO
    #     """
    #     Get ensembler classes to combine multiple predictions
    #     Needs to be overwritten in subclasses!
    #     """
    #     raise NotImplementedError

    # @classmethod
    # def get_predictor(
    #     cls,
    #     plan: Dict,
    #     models: Sequence[LightningBaseModule],
    #     num_tta_transforms: int = None,
    #     **kwargs,
    # ) -> Type[Predictor]: # TODO
    #     """
    #     Get predictor
    #     Needs to be overwritten in subclasses!
    #     """
    #     raise NotImplementedError

    # def sweep(
    #     self,
    #     cfg: dict,
    #     save_dir: os.PathLike,
    #     train_data_dir: os.PathLike,
    #     case_ids: Sequence[str],
    #     run_prediction: bool = True,
    # ) -> Dict[str, Any]:  # TODO
    #     """
    #     Sweep parameters to find the best predictions
    #     Needs to be overwritten in subclasses!

    #     Args:
    #         cfg: config used for training
    #         save_dir: save dir used for training
    #         train_data_dir: directory where preprocessed training/validation
    #             data is located
    #         case_ids: case identifies to prepare and predict
    #         run_prediction: predict cases
    #         **kwargs: keyword arguments passed to predict function
    #     """
    #     raise NotImplementedError

    def configure_callbacks(self):
        callbacks = super().configure_callbacks()
        callbacks.append(EpochTimerCallback())

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
