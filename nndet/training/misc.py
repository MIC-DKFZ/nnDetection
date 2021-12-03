import time

from loguru import logger
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback


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

    def on_train_epoch_start(
        self,
        trainer,
        pl_module: LightningModule,
    ) -> None:
        self.train_epoch_tic = time.time()
        return super().on_train_epoch_start(trainer, pl_module)

    def on_train_epoch_end(
        self,
        trainer,
        pl_module: LightningModule,
    ) -> None:
        self.train_epoch_toc = time.time()
        logger.info(
            f"Train epoch {trainer.current_epoch} took "
            f"{int(self.train_epoch_toc - self.train_epoch_tic)} s"
        )
        return super().on_train_epoch_end(trainer, pl_module)

    def on_validation_epoch_start(
        self,
        trainer,
        pl_module: LightningModule,
    ) -> None:
        self.val_epoch_tic = time.time()
        return super().on_validation_epoch_start(trainer, pl_module)

    def on_validation_epoch_end(
        self,
        trainer,
        pl_module: LightningModule,
    ) -> None:
        self.val_epoch_toc = time.time()
        logger.info(
            f"Val epoch {trainer.current_epoch} took "
            f"{int(self.val_epoch_toc - self.val_epoch_tic)} s"
        )
        return super().on_validation_epoch_end(trainer, pl_module)
