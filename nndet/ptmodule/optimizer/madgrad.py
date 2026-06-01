# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import nndet
from nndet.ptmodule.optimizer import OPTIMIZER_REGISTRY
from nndet.training.learning_rate import LinearWarmupPolyLR
from nndet.training.optimizer import get_params_no_wd_on_norm


@OPTIMIZER_REGISTRY.register
class MadgradLWPoly:
    """
    Madgrad Optimizer Mixin

    Please refer to the official
    `repo <https://github.com/facebookresearch/madgrad>`_ for more info.
    """

    @classmethod
    def configure_optimizers(
        cls,
        module: "nndet.ptmodule.module.LightningBaseModule",
    ):
        try:
            from madgrad import MADGRAD
        except ImportError:
            raise ImportError(
                "madgrad needs to be installed to run this module."
                "Please refer to https://github.com/facebookresearch/madgrad"
                "to install it"
            )
        trainer_cfg = module.trainer_cfg

        # configure optimizer
        logger.info(
            f"Running: initial_lr {trainer_cfg['initial_lr']} "
            f"weight_decay {trainer_cfg['weight_decay']} "
            f"MADGRAD with momentum {trainer_cfg['momentum']}"
        )
        wd_groups = get_params_no_wd_on_norm(module, weight_decay=trainer_cfg["weight_decay"])
        optimizer = MADGRAD(
            wd_groups,
            trainer_cfg["initial_lr"],
            weight_decay=trainer_cfg["weight_decay"],
            momentum=trainer_cfg["momentum"],
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
