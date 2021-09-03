import torch
from loguru import logger

from nndet.training.learning_rate import LinearWarmupPolyLR
from nndet.training.optimizer import get_params_no_wd_on_norm


class AdamWDefaultMixin:
    def configure_optimizers(self):
        # configure optimizer
        logger.info(
            f"Running: initial_lr {self.trainer_cfg['initial_lr']} "
            f"weight_decay {self.trainer_cfg['weight_decay']} "
            f"AdamW"
        )
        wd_groups = get_params_no_wd_on_norm(
            self, weight_decay=self.trainer_cfg["weight_decay"]
        )
        optimizer = torch.optim.AdamW(
            wd_groups,
            self.trainer_cfg["initial_lr"],
            weight_decay=self.trainer_cfg["weight_decay"],
        )

        # configure lr scheduler
        num_iterations = (
            self.train_epochs * self.trainer_cfg["num_train_batches_per_epoch"]
        )
        scheduler = LinearWarmupPolyLR(
            optimizer=optimizer,
            warm_iterations=self.trainer_cfg["warm_iterations"],
            warm_lr=self.trainer_cfg["warm_lr"],
            poly_gamma=self.trainer_cfg["poly_gamma"],
            num_iterations=num_iterations,
        )
        return [optimizer], {"scheduler": scheduler, "interval": "step"}
