# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Callable, Optional, Union, cast

import torch
from lightning_fabric.utilities.types import LRScheduler
from loguru import logger
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.types import LRSchedulerConfig
from torch.optim.lr_scheduler import _LRScheduler

from nndet.training.learning_rate import CycleLinear

_AVG_FN = Callable[[torch.Tensor, torch.Tensor, torch.LongTensor], torch.FloatTensor]


class BaseSWA(StochasticWeightAveraging):
    def __init__(
        self,
        swa_epoch_start: int,
        avg_fn: Optional[_AVG_FN] = None,
        device: Optional[Union[torch.device, str]] = torch.device("cpu"),
        update_statistics: Optional[bool] = False,
    ):
        """
        New Base Class for Stochastic Weighted Averaging

        Args:
            swa_epoch_start: Epoch to start SWA weight saving.
            avg_fn: Function to average saved weights. Defaults to None.
            device: Device to save averaged model. Defaults to
                torch.device("cpu").
            update_statistics: Perform a final update of the normalization
                layers. Defaults to None.

        Notes: Does not support updating of norm weights after training
        """
        super().__init__(
            swa_epoch_start=swa_epoch_start,
            swa_lrs=[0.1],
            annealing_epochs=10,
            annealing_strategy="cos",
            avg_fn=avg_fn,
            device=device,
        )
        self._swa_lrs = None
        self.update_statistics = update_statistics
        self._init_n_averaged = 0
        logger.info(f"Initialize SWA with swa epoch start {self.swa_start}")
        logger.warning(
            "SWA in nnDetection is somewhat experimental and result "
            "in unexpected behavior in multi GPU scenrios or when the "
            "state needs to be restored. Please prefer V2 or newer "
            "model for your experiments"
        )

    def pl_module_contains_batch_norm(self, pl_module: "pl.LightningModule"):  # noqa: F821
        if self.update_statistics:
            raise NotImplementedError("Updating the statistis of the " "normalization layer is not suported yet.")
        else:
            return self.update_statistics

    def on_train_epoch_start(
        self,
        trainer: "pl.Trainer",  # noqa: F821
        pl_module: "pl.LightningModule",  # noqa: F821
    ):
        """
        Repalce current lr scheduler with SWA scheduler
        """
        if trainer.current_epoch == self.swa_start:
            optimizer = trainer.optimizers[0]

            # move average model to request device.
            self._average_model = self._average_model.to(self._device or pl_module.device)

            _scheduler = self.get_swa_scheduler(optimizer)

            # handle scheduler
            if not isinstance(_scheduler, dict):
                _scheduler = {"scheduler": _scheduler}
            self._swa_scheduler = cast(
                LRScheduler,
                _scheduler.pop("scheduler"),
            )  # retrieve scheduler from dict
            if self._scheduler_state is not None:
                # Restore scheduler state from checkpoint
                self._swa_scheduler.load_state_dict(self._scheduler_state)

            # handle config
            _scheduler_config_defaults = {
                "name": None,
                "interval": "epoch",
                "frequency": 1,
                "reduce_on_plateau": False,
                "monitor": None,
                "strict": True,
            }
            _scheduler_config_defaults.update(_scheduler)
            self._scheduler_config = _scheduler_config_defaults

            swa_lr_scheduler_config = LRSchedulerConfig(scheduler=self._swa_scheduler, **self._scheduler_config)

            if trainer.lr_scheduler_configs:
                lr_scheduler_config = trainer.lr_scheduler_configs[0]
                rank_zero_warn(f"Swapping lr_scheduler {lr_scheduler_config} for {swa_lr_scheduler_config}")
                trainer.lr_scheduler_configs[0] = swa_lr_scheduler_config
            else:
                trainer.lr_scheduler_configs.append(swa_lr_scheduler_config)

            self.n_averaged = torch.tensor(self._init_n_averaged, dtype=torch.long, device=pl_module.device)

        if self.swa_start <= trainer.current_epoch <= self.swa_end:
            self.update_parameters(self._average_model, pl_module, self.n_averaged, self.avg_fn)

        if trainer.current_epoch == self.swa_end + 1:
            raise NotImplementedError("This should never happen (yet)")

    @abstractmethod
    def get_swa_scheduler(self, optimizer) -> Union[_LRScheduler, dict]:
        """
        Generate LR scheduler for SWA

        Args:
            optimizer: optimizer to wrap

        Returns:
            Union[_LRScheduler, dict]: If a lr scheduler is returned it will
                be stepped once per epoch. Can also return a whole config of
                the scheduler to customize steps.
        """
        raise NotImplementedError


class SWACycleLinear(BaseSWA):
    def __init__(
        self,
        swa_epoch_start: int,
        cycle_initial_lr: float,
        cycle_final_lr: float,
        num_iterations_per_epoch: int,
        avg_fn: Optional[_AVG_FN] = None,
        device: Optional[Union[torch.device, str]] = torch.device("cpu"),
        update_statistics: Optional[bool] = None,
    ):
        """
        SWA based on :class:`CycleLinear`

        Args:
            swa_epoch_start: Epoch to start SWA weight saving.
            cycle_initial_lr: initial learning rate of cycle
            cycle_final_lr: final learning rate of cycle
            num_iterations_per_epoch: number of train iterations per epoch
            avg_fn: Function to average saved weights. Defaults to None.
            device: Device to save averaged model. Defaults to
                torch.device("cpu").
            update_statistics: Perform a final update of the normalization
                layers. Defaults to None.
        """
        super().__init__(
            swa_epoch_start=swa_epoch_start,
            avg_fn=avg_fn,
            device=device,
            update_statistics=update_statistics,
        )
        self.cycle_initial_lr = cycle_initial_lr
        self.cycle_final_lr = cycle_final_lr
        self.num_iterations_per_epoch = num_iterations_per_epoch

    def get_swa_scheduler(self, optimizer) -> Union[_LRScheduler, dict]:
        return {
            "scheduler": CycleLinear(
                optimizer=optimizer,
                cycle_num_iterations=self.num_iterations_per_epoch,
                cycle_initial_lr=self.cycle_initial_lr,
                cycle_final_lr=self.cycle_final_lr,
            ),
            "interval": "step",
        }
