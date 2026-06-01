# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import nndet
from nndet.ptmodule.optimizer import OPTIMIZER_REGISTRY
from nndet.training.learning_rate import LinearWarmupPolyLR
from nndet.training.optimizer import get_params_no_wd_on_norm


@OPTIMIZER_REGISTRY.register
class RangerLWPoly:
    """
    Ranger Optimizer Mixin

    Please refer to https://github.com/jettify/pytorch-optimizer for more info.
    """

    @classmethod
    def configure_optimizers(
        cls,
        module: "nndet.ptmodule.module.LightningBaseModule",
    ):
        try:
            import torch_optimizer as optim
        except ImportError:
            raise ImportError(
                "torch_optimizer needs to be installed to run this module."
                "Please refer to https://github.com/jettify/pytorch-optimizer"
                "to install it"
            )
        trainer_cfg = module.trainer_cfg

        # configure optimizer
        logger.info(
            f"Running: initial_lr {trainer_cfg['initial_lr']} " f"weight_decay {trainer_cfg['weight_decay']} " f"Ranger"
        )
        wd_groups = get_params_no_wd_on_norm(module, weight_decay=trainer_cfg["weight_decay"])
        optimizer = optim.Ranger(
            wd_groups,
            trainer_cfg["initial_lr"],
            weight_decay=trainer_cfg["weight_decay"],
        )

        # configure lr scheduler
        num_iterations = module.train_epochs * trainer_cfg["num_train_batches_per_epoch"]
        scheduler = LinearWarmupPolyLR(
            optimizer=optimizer,
            warm_iterations=trainer_cfg["warm_iterations"],
            warm_lr=trainer_cfg["warm_lr"],
            poly_gamma=trainer_cfg["poly_gamma"],
            num_iterations=num_iterations,
        )
        return [optimizer], {"scheduler": scheduler, "interval": "step"}


@OPTIMIZER_REGISTRY.register
class Ranger21:
    """
    Ranger21 Optimizer Mixin

    Please refer to https://github.com/lessw2020/Ranger21 for more info.
    """

    @classmethod
    def configure_optimizers(
        cls,
        module: "nndet.ptmodule.module.LightningBaseModule",
    ):
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
        trainer_cfg = module.trainer_cfg

        # configure optimizer
        logger.info(
            f"Running: initial_lr {trainer_cfg['initial_lr']} "
            f"weight_decay {trainer_cfg['weight_decay']} "
            f"Ranger21"
        )
        optimizer = Ranger21(
            module.parameters(),
            lr=trainer_cfg["initial_lr"],
            weight_decay=trainer_cfg["weight_decay"],
            use_cheb=False,
            lookahead_active=True,
            normloss_active=True,
            normloss_factor=6e-4,
            use_adaptive_gradient_clipping=True,
            agc_clipping_value=0.01,
            use_madgrad=False,
            warmdown_active=True,
            num_warmup_iterations=None,
            num_epochs=module.train_epochs,
            num_batches_per_epoch=trainer_cfg["num_train_batches_per_epoch"],
            warmup_pct_default=0.3,
            using_gc=True,
        )
        optimizer.show_settings()
        return optimizer
