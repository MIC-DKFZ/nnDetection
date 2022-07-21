# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import nndet
from nndet.ptmodule.optimizer import OPTIMIZER_REGISTRY
from nndet.training.learning_rate import LinearWarmupPolyLR
from nndet.training.optimizer import get_params_no_wd_on_norm


@OPTIMIZER_REGISTRY.register
class RAdamLWPoly:
    """
    RAdam Optimizer Mixin

    Please refer to the following
    `repo <https://github.com/jettify/pytorch-optimizer>`_ for more info.
    """

    @classmethod
    def configure_optimizers(
        cls,
        module: "nndet.ptmodule.module.LightningBaseModuleType",
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
            f"Running: initial_lr {trainer_cfg['initial_lr']} "
            f"weight_decay {trainer_cfg['weight_decay']} "
            f"RAdam"
        )
        wd_groups = get_params_no_wd_on_norm(
            module, weight_decay=trainer_cfg["weight_decay"]
        )
        optimizer = optim.RAdam(
            wd_groups,
            lr=trainer_cfg["initial_lr"],
            weight_decay=trainer_cfg["weight_decay"],
        )

        # configure lr scheduler
        num_iterations = (
            module.train_epochs * trainer_cfg["num_train_batches_per_epoch"]
        )
        scheduler = LinearWarmupPolyLR(
            optimizer=optimizer,
            warm_iterations=trainer_cfg["warm_iterations"],
            warm_lr=trainer_cfg["warm_lr"],
            poly_gamma=trainer_cfg["poly_gamma"],
            num_iterations=num_iterations,
        )
        return [optimizer], {"scheduler": scheduler, "interval": "step"}
