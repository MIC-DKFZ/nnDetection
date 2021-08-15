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

import os
from time import time
from typing import Any, Callable, Dict, Optional, Sequence, Hashable, Type, TypeVar

import torch
import pytorch_lightning as pl
from pytorch_lightning.core.memory import ModelSummary
from loguru import logger

from nndet.io.load import save_txt
from nndet.inference.predictor import Predictor


class LightningBaseModule(pl.LightningModule):
    def __init__(self,
                 model_cfg: dict,
                 trainer_cfg: dict,
                 plan: dict,
                 **kwargs
                 ):
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

        self.model = self.from_config_plan(
            model_cfg=self.model_cfg,
            plan_arch=self.plan["architecture"],
            plan_anchors=self.plan["anchors"],
        )

        self.example_input_array_shape = (
            1, plan["architecture"]["in_channels"], *plan["patch_size"],
            )

        self.epoch_start_tic = 0
        self.epoch_end_toc = 0

    @property
    def max_epochs(self):
        """
        Number of epochs to train
        """
        return self.trainer_cfg["max_num_epochs"]

    def on_epoch_start(self) -> None:
        """
        Save time
        """
        self.epoch_start_tic = time()
        return super().on_epoch_start()
    
    def validation_epoch_end(self, validation_step_outputs):
        """
        Print time of epoch
        (needed for cluster where progress bar is deactivated)
        """
        self.epoch_end_toc = time()
        logger.info(f"This epoch took {int(self.epoch_end_toc - self.epoch_start_tic)} s")
        return super().validation_epoch_end(validation_step_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used to generate summary
        Do not(!) use this for inference. This will only forward
        the input through the network which does not include
        detection spcific postprocessing!
        """
        return self.model(x)

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

    @classmethod
    def from_config_plan(cls,
                         model_cfg: dict,
                         plan_arch: dict,
                         plan_anchors: dict,
                         log_num_anchors: str = None,
                         **kwargs,
                         ):
        """
        Used to generate the model
        """
        raise NotImplementedError

    @staticmethod
    def get_ensembler_cls(key: Hashable, dim: int) -> Callable:
        """
        Get ensembler classes to combine multiple predictions
        Needs to be overwritten in subclasses!
        """
        raise NotImplementedError

    @classmethod
    def get_predictor(cls,
                      plan: Dict,
                      models: Sequence[LightningBaseModule],
                      num_tta_transforms: int = None,
                      **kwargs
                      ) -> Type[Predictor]:
        """
        Get predictor
        Needs to be overwritten in subclasses!
        """
        raise NotImplementedError

    def sweep(self,
              cfg: dict,
              save_dir: os.PathLike,
              train_data_dir: os.PathLike,
              case_ids: Sequence[str],
              run_prediction: bool = True,
              ) -> Dict[str, Any]:
        """
        Sweep parameters to find the best predictions
        Needs to be overwritten in subclasses!

        Args:
            cfg: config used for training
            save_dir: save dir used for training
            train_data_dir: directory where preprocessed training/validation
                data is located
            case_ids: case identifies to prepare and predict
            run_prediction: predict cases
            **kwargs: keyword arguments passed to predict function
        """
        raise NotImplementedError


class LightningBaseModuleSWA(LightningBaseModule):
    @property
    def max_epochs(self):
        """
        Number of epochs to train
        """
        return self.trainer_cfg["max_num_epochs"] + self.trainer_cfg["swa_epochs"]

    def configure_callbacks(self):
        from nndet.training.swa import SWACycleLinear

        callbacks = []
        callbacks.append(
            SWACycleLinear(
                swa_epoch_start=self.trainer_cfg["max_num_epochs"],
                cycle_initial_lr=self.trainer_cfg["initial_lr"] / 10.,
                cycle_final_lr=self.trainer_cfg["initial_lr"] / 1000.,
                num_iterations_per_epoch=self.trainer_cfg["num_train_batches_per_epoch"],
                )
            )
        return callbacks


LightningBaseModuleType = TypeVar('LightningBaseModuleType', bound=LightningBaseModule)
