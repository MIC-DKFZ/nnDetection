import time

from loguru import logger
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback

from nndet.training.ema import EMA


class EpochTimerCallback(Callback):
    def __init__(self) -> None:
        """
        Simple callback to print epoch times.
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

    def on_train_start(self, *args, **kwargs) -> None:
        self.train_tic = time.time()

    def on_train_end(self, *args, **kwargs) -> None:
        train_toc = time.time()
        measured_time = train_toc - self.train_tic
        logger.info(f"### Training took {(measured_time / 3600):.2f} hours. ###")

    def on_train_epoch_start(
        self,
        trainer,
        pl_module: LightningModule,
    ) -> None:
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
