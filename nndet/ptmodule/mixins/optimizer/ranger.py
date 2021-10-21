from loguru import logger

from nndet.training.learning_rate import LinearWarmupPolyLR
from nndet.training.optimizer import get_params_no_wd_on_norm


class RangerDefaultMixin:
    """
    Ranger Optimizer Mixin
    
    Please refer to the following
    `repo <https://github.com/jettify/pytorch-optimizer>`_ for more info.
    """
    def configure_optimizers(self):
        try:
            import torch_optimizer as optim
        except ImportError:
            raise ImportError(
                "torch_optimizer needs to be installed to run this module."
                "Please refer to https://github.com/jettify/pytorch-optimizer"
                "to install it"
            )

        # configure optimizer
        logger.info(
            f"Running: initial_lr {self.trainer_cfg['initial_lr']} "
            f"weight_decay {self.trainer_cfg['weight_decay']} "
            f"Ranger"
        )
        wd_groups = get_params_no_wd_on_norm(
            self, weight_decay=self.trainer_cfg["weight_decay"]
        )
        optimizer = optim.Ranger(
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


class Ranger21DefaultMixin:
    """
    Ranger21 Optimizer Mixin
    
    Please refer to the following
    `repo <https://github.com/lessw2020/Ranger21>`_ for more info.
    """
    def configure_optimizers(self):
        """
        Experimental Settings
        """
        try:
            from ranger21 import Ranger21
        except ImportError:
            raise ImportError(
                "ranger21 needs to be installed to run this module."
                "Please refer to https://github.com/lessw2020/Ranger21"
                "to install it"
            )

        # configure optimizer
        logger.info(
            f"Running: initial_lr {self.trainer_cfg['initial_lr']} "
            f"weight_decay {self.trainer_cfg['weight_decay']} "
            f"Ranger21"
        )
        optimizer = Ranger21(
            self.parameters(),
            lr=self.trainer_cfg["initial_lr"],
            weight_decay=self.trainer_cfg["weight_decay"],
            use_cheb=False,
            lookahead_active=True,
            normloss_active=True,
            normloss_factor=6e-4,
            use_adaptive_gradient_clipping=True,
            agc_clipping_value=0.01,
            use_madgrad=False,
            warmdown_active=True,
            num_warmup_iterations=None,
            num_epochs=self.train_epochs,
            num_batches_per_epoch=self.trainer_cfg["num_train_batches_per_epoch"],
            warmup_pct_default=0.3,
            using_gc=True,
        )
        optimizer.show_settings()
        return optimizer
