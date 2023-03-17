# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

import nndet
from nndet.ptmodule.optimizer import OPTIMIZER_REGISTRY
from nndet.training.learning_rate import LinearWarmupPolyLR
from nndet.training.optimizer import get_params_no_wd_on_norm


@OPTIMIZER_REGISTRY.register
class AdamWLWPoly:
    """
    AdamW Optimizer Mixin
    """

    @classmethod
    def configure_optimizers(
        cls,
        module: "nndet.ptmodule.module.LightningBaseModule",
    ):
        """
        Configure optimizer and scheduler
        Base configuration is `AdamW` with `LinearWarmup` and `PolyLR` learning
        rate schedule. Configuration is done via the config file.

        module: module to configure with `trainer_cfg`

            ``"initial_lr"`` float
                learning rate *after* warmup

            ``"weight_decay"`` float
                weight decay passed to optimizer

            ``"beta1"`` float
                Adam first beta param. See PyTorch docs for more info.

            ``"beta2"`` float
                Adam second beta param. See PyTorch docs for more info.

            ``"eps"`` float
                Adam eps param. See PyTorch docs for more info.

            ``"amsgrad"`` bool
                Enable amsgrad. See PyTorch docs for more info.

            ``"num_train_batches_per_epoch"`` int
                number of batches per epoch

            ``"warm_iterations"`` int
                number of iterations to run warm up

            ``"warm_lr"`` float
                learning rate to start warming up from

            ``"poly_gamma"`` float
                gamma term passed to PolyLR

        """
        trainer_cfg = module.trainer_cfg

        # configure optimizer
        logger.info(
            f"Running: initial_lr {trainer_cfg['initial_lr']} " f"weight_decay {trainer_cfg['weight_decay']} " f"AdamW"
        )
        wd_groups = get_params_no_wd_on_norm(module, weight_decay=trainer_cfg["weight_decay"])
        betas = (trainer_cfg["beta1"], trainer_cfg["beta2"])
        optimizer = torch.optim.AdamW(
            wd_groups,
            trainer_cfg["initial_lr"],
            weight_decay=trainer_cfg["weight_decay"],
            betas=betas,
            eps=trainer_cfg["eps"],
            amsgrad=trainer_cfg["amsgrad"],
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
