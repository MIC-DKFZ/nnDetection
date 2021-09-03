from loguru import logger

from nndet.training.learning_rate import LinearWarmupPolyLR
from nndet.training.optimizer import get_params_no_wd_on_norm


class MadgradDefaultMixin:
    def configure_optimizers(self):
        try:
            from madgrad import MADGRAD
        except ImportError:
            raise ImportError(
                "madgrad needs to be installed to run this module."
                "Please refer to https://github.com/facebookresearch/madgrad"
                "to install it"
            )

        # configure optimizer
        logger.info(
            f"Running: initial_lr {self.trainer_cfg['initial_lr']} "
            f"weight_decay {self.trainer_cfg['weight_decay']} "
            f"MADGRAD with momentum {self.trainer_cfg['momentum']}"
        )
        wd_groups = get_params_no_wd_on_norm(
            self, weight_decay=self.trainer_cfg["weight_decay"]
        )
        optimizer = MADGRAD(
            wd_groups,
            self.trainer_cfg["initial_lr"],
            weight_decay=self.trainer_cfg["weight_decay"],
            momentum=self.trainer_cfg["momentum"],
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
