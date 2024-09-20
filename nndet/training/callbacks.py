# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import time
from collections import deque
from typing import Hashable

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback

from nndet.training.ema import EMA


class WeightsNaNError(Exception):
    """
    Custom exception if NaN weights are found during training.
    """

    pass


class LossNaNError(Exception):
    """
    Custom exception if NaN loss is found during training.
    """

    pass


class LowPerformanceError(Exception):
    """
    Custom exception if performance is lower than a certain threshold.
    """

    pass


class CheckWeightsNaN(Callback):
    def on_train_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        found_nan = False
        for name, param in pl_module.named_parameters():
            if torch.isnan(param).any():
                found_nan = True
                logger.error(f"Found NaN parameter in module {name}, aborting training.")

        if found_nan:
            raise WeightsNaNError("Found NaN parameter in module, aborting training.")

        return super().on_train_epoch_end(trainer, pl_module)


class EpochTimerCallback(Callback):
    def __init__(self) -> None:
        """
        Simple callback to print epoch times and epoch info.
        This is useful in cluster environmens where the progress bar is
        deactivated
        """
        super().__init__()
        self.train_epoch_tic = 0
        self.train_epoch_toc = 0
        self.val_epoch_tic = 0
        self.val_epoch_toc = 0

        self.train_time_ema = EMA(beta=0.9, bias_correction=True)
        self.val_time_ema = EMA(beta=0.9, bias_correction=True)

        self.train_tic = 0

    def on_fit_start(self, *args, **kwargs) -> None:
        self.train_tic = time.time()

    def on_fit_end(self, *args, **kwargs) -> None:
        train_toc = time.time()
        measured_time = train_toc - self.train_tic
        logger.info(f"### Training took {(measured_time / 3600):.2f} hours. ###")

    def on_train_epoch_start(
        self,
        trainer,
        pl_module: LightningModule,
    ) -> None:
        logger.info(f"+++ Epoch {trainer.current_epoch} +++")
        self.train_epoch_tic = time.time()
        return super().on_train_epoch_start(trainer, pl_module)

    def on_validation_epoch_start(
        self,
        trainer,
        pl_module: LightningModule,
    ) -> None:
        if self.train_epoch_tic > 0:
            self.train_epoch_toc = time.time()
            train_time = int(self.train_epoch_toc - self.train_epoch_tic)
            self.train_time_ema.add(train_time)
            logger.info(
                f"Train epoch {trainer.current_epoch} took "
                f"{train_time:.2f} s and train EMA is {self.train_time_ema.get():.2f} s"
            )

        self.val_epoch_tic = time.time()
        return super().on_validation_epoch_start(trainer, pl_module)

    def on_validation_epoch_end(
        self,
        trainer,
        pl_module: LightningModule,
    ) -> None:
        self.val_epoch_toc = time.time()
        val_time = int(self.val_epoch_toc - self.val_epoch_tic)
        self.val_time_ema.add(val_time)

        logger.info(
            f"Val epoch {trainer.current_epoch} took "
            f"{val_time:.2f} s and val time EMA is {self.val_time_ema.get():.2f} s"
        )
        return super().on_validation_epoch_end(trainer, pl_module)

    def on_sanity_check_start(
        self,
        trainer,
        pl_module: LightningModule,
    ) -> None:
        logger.info("+++ Sanity Check +++")
        return super().on_train_epoch_start(trainer, pl_module)


class CheckLowPerformance(Callback):
    def __init__(
        self,
        threshold: float,
        wait_epochs: int,
        avg_epochs: int,
        monitor_key: Hashable,
    ) -> None:
        """
        Callback to check if the performance is lower than a certain threshold.
        If the performance is lower, the training is aborted.

        Args:
            threshold: The threshold to check the performance against.
            wait_epochs: The number of epochs to wait before checking the
                performance.
            avg_epochs: The number of epochs to average the performance over.
            monitor_key: The key of the metric to monitor.
        """
        super().__init__()
        self.threshold = threshold
        self.wait_epochs = wait_epochs
        self.avg_epochs = avg_epochs
        self.monitor_key = monitor_key

        self.monitored_values = deque([], maxlen=avg_epochs)

    def on_train_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        module_logs = trainer.callback_metrics
        self.monitored_values.append(float(module_logs[self.monitor_key]))

        print(f"Performance: {module_logs[self.monitor_key]}")
        print(self.monitored_values)

        if trainer.current_epoch == (self.wait_epochs - 1):
            avg_performance = sum(self.monitored_values) / len(self.monitored_values)
            if avg_performance < self.threshold:
                raise LowPerformanceError(
                    f"Performance {avg_performance} is lower than threshold {self.threshold} after {self.wait_epochs}."
                )
        return None
