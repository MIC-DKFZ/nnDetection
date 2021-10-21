import torch
from loguru import logger

from nndet.training.learning_rate import LinearWarmupPolyLR
from nndet.training.optimizer import get_params_no_wd_on_norm


class SGDDefaultMixin:
    """
    SGD Optimizer Mixin
    """
    def configure_optimizers(self):
        """
        Configure optimizer and scheduler
        Base configuration is `SGD` with `LinearWarmup` and `PolyLR` learning rate
        schedule. Configuration is done via the config file.

        Configurations keys:
            - `initial_lr` (float): learning rate *after* warmup
            - `weight_decay` (float): weight decay passed to optimzier
            - `sgd_momentum` (float): momentum term passed to optimizer
            - `sgd_nesterov` (bool): passed to optimizer
            - `num_train_batches_per_epoch` (int): number of batches per epoch
            - `warm_iterations` (int): number of iterations to runw warm up
            - `warm_lr` (float): learning rate to start warming up from
            - `poly_gamma` (float): gamma term passed to PolyLR
        """
        # configure optimizer
        logger.info(
            f"Running: initial_lr {self.trainer_cfg['initial_lr']} "
            f"weight_decay {self.trainer_cfg['weight_decay']} "
            f"SGD with momentum {self.trainer_cfg['sgd_momentum']} and "
            f"nesterov {self.trainer_cfg['sgd_nesterov']}"
        )
        wd_groups = get_params_no_wd_on_norm(
            self, weight_decay=self.trainer_cfg["weight_decay"]
        )
        optimizer = torch.optim.SGD(
            wd_groups,
            self.trainer_cfg["initial_lr"],
            weight_decay=self.trainer_cfg["weight_decay"],
            momentum=self.trainer_cfg["sgd_momentum"],
            nesterov=self.trainer_cfg["sgd_nesterov"],
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
