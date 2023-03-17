# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

import nndet
from nndet.ptmodule.optimizer import OPTIMIZER_REGISTRY
from nndet.training.learning_rate import LinearWarmupPolyLR, PolyLR
from nndet.training.optimizer import get_params_no_wd_on_norm


@OPTIMIZER_REGISTRY.register
class SGDLWPoly:
    """
    SGD Optimizer Mixin with Linear Warmup into PolyLR
    """

    @classmethod
    def configure_optimizers(
        cls,
        module: "nndet.ptmodule.module.LightningBaseModule",
    ):
        """
        Configure optimizer and scheduler
        Base configuration is `SGD` with `LinearWarmup` and `PolyLR` learning rate
        schedule. Configuration is done via the config file.

        module: module to configure with `trainer_cfg`

            ``"initial_lr"`` float
                learning rate *after* warmup

            ``"weight_decay"`` float
                weight decay passed to optimzier

            ``"sgd_momentum"`` float
                momentum term passed to optimizer

            ``"sgd_nesterov"`` bool
                passed to optimizer

            ``"num_train_batches_per_epoch"`` int
                number of batches per epoch

            ``"warm_iterations"`` int
                number of iterations to runw warm up

            ``"warm_lr"`` float
                learning rate to start warming up from

            ``"poly_gamma"`` float
                gamma term passed to PolyLR

        """
        trainer_cfg = module.trainer_cfg

        # configure optimizer
        logger.info(
            f"Running {cls.__name__}: "
            f"initial_lr {trainer_cfg['initial_lr']} "
            f"weight_decay {trainer_cfg['weight_decay']} "
            f"SGD with momentum {trainer_cfg['sgd_momentum']} and "
            f"nesterov {trainer_cfg['sgd_nesterov']}"
        )
        wd_groups = get_params_no_wd_on_norm(module, weight_decay=trainer_cfg["weight_decay"])
        optimizer = torch.optim.SGD(
            wd_groups,
            trainer_cfg["initial_lr"],
            weight_decay=trainer_cfg["weight_decay"],
            momentum=trainer_cfg["sgd_momentum"],
            nesterov=trainer_cfg["sgd_nesterov"],
            # foreach=True,
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
class SGDPoly:
    """
    SGD Optimizer Mixin with PolyLR
    """

    @classmethod
    def configure_optimizers(
        cls,
        module: "nndet.ptmodule.module.LightningBaseModule",
    ):
        """
        Configure optimizer and scheduler
        Base configuration is `SGD` with `PolyLR` learning rate
        schedule. Configuration is done via the config file.

        module: module to configure with `trainer_cfg`

            ``"initial_lr"`` float
                learning rate *after* warmup

            ``"weight_decay"`` float
                weight decay passed to optimzier

            ``"sgd_momentum"`` float
                momentum term passed to optimizer

            ``"sgd_nesterov"`` bool
                passed to optimizer

            ``"num_train_batches_per_epoch"`` int
                number of batches per epoch

            ``"poly_gamma"`` float
                gamma term passed to PolyLR
        """
        trainer_cfg = module.trainer_cfg

        # configure optimizer
        logger.info(
            f"Running {cls.__name__}: "
            f"initial_lr {trainer_cfg['initial_lr']} "
            f"weight_decay {trainer_cfg['weight_decay']} "
            f"SGD with momentum {trainer_cfg['sgd_momentum']} and "
            f"nesterov {trainer_cfg['sgd_nesterov']}"
        )
        wd_groups = get_params_no_wd_on_norm(module, weight_decay=trainer_cfg["weight_decay"])
        optimizer = torch.optim.SGD(
            wd_groups,
            trainer_cfg["initial_lr"],
            weight_decay=trainer_cfg["weight_decay"],
            momentum=trainer_cfg["sgd_momentum"],
            nesterov=trainer_cfg["sgd_nesterov"],
            # foreach=True,
        )

        # configure lr scheduler
        num_iterations = module.train_epochs * trainer_cfg["num_train_batches_per_epoch"]
        scheduler = PolyLR(
            optimizer=optimizer,
            poly_gamma=trainer_cfg["poly_gamma"],
            num_iterations=num_iterations,
        )
        return [optimizer], {"scheduler": scheduler, "interval": "step"}


@OPTIMIZER_REGISTRY.register
class TwoStageSGDLWPoly:
    """
    SGD Optimizer Mixin
    """

    @classmethod
    def configure_optimizers(
        cls,
        module: "nndet.ptmodule.module.LightningBaseModule",
    ):
        """
        Configure optimizer and scheduler for two stage detector
        Base configuration is `SGD` with `LinearWarmup` and `PolyLR` learning rate
        schedule. Configuration is done via the config file.

        module: module to configure with `trainer_cfg`

            ``"rpn_initial_lr"`` float
                learning rate *after* warmup

            ``"roi_initial_lr"`` float
                learning rate *after* warmup

            ``"rpn_sgd_momentum"`` float
                momentum term passed to optimizer

            ``"roi_sgd_momentum"`` float
                momentum term passed to optimizer

            ``"weight_decay"`` float
                weight decay passed to optimzier

            ``"sgd_nesterov"`` bool
                passed to optimizer

            ``"num_train_batches_per_epoch"`` int
                number of batches per epoch

            ``"warm_iterations"`` int
                number of iterations to runw warm up

            ``"warm_lr"`` float
                learning rate to start warming up from

            ``"poly_gamma"`` float
                gamma term passed to PolyLR

        Warning::
            This optimizer only works with two stage models where the
            Region proposal network is saved into `model.rpn` and
            the RoI module is saved into `model.roi_module`. No other
            parameters will be updated by this optimizer!
        """
        trainer_cfg = module.trainer_cfg

        if not hasattr(module, "model"):
            raise ValueError("Detector needs to be saved in 'model' param of lightning module!")
        if not hasattr(module.model, "rpn"):
            raise ValueError("RPN needs to be saved in 'model.rpn' param of detector!")
        if not hasattr(module.model, "roi_module"):
            raise ValueError("RoI needs to be saved in 'model.roi_module' param of detector!")

        rpn_initial_lr = trainer_cfg["rpn_initial_lr"]
        roi_initial_lr = trainer_cfg["roi_initial_lr"]
        rpn_sgd_momentum = trainer_cfg["rpn_sgd_momentum"]
        roi_sgd_momentum = trainer_cfg["roi_sgd_momentum"]

        # configure optimizer
        logger.info(
            "Running Optimizer: "
            f"RPN initial lr {rpn_initial_lr}, momentum {rpn_sgd_momentum} "
            f"RoI initial lr {roi_initial_lr}, momentum {roi_sgd_momentum} "
            f"All weight decay {trainer_cfg['weight_decay']}, nesterov {trainer_cfg['sgd_nesterov']} "
        )

        param_groups = []
        # configure RPN
        rpn_wd_groups = get_params_no_wd_on_norm(module.model.rpn, weight_decay=trainer_cfg["weight_decay"])
        for idx in range(len(rpn_wd_groups)):
            rpn_wd_groups[idx]["lr"] = rpn_initial_lr
            rpn_wd_groups[idx]["momentum"] = rpn_sgd_momentum
        param_groups.extend(rpn_wd_groups)

        # configure RoI
        roi_wd_groups = get_params_no_wd_on_norm(module.model.roi_module, weight_decay=trainer_cfg["weight_decay"])
        for idx in range(len(roi_wd_groups)):
            roi_wd_groups[idx]["lr"] = roi_initial_lr
            roi_wd_groups[idx]["momentum"] = roi_sgd_momentum
        param_groups.extend(roi_wd_groups)

        optimizer = torch.optim.SGD(
            param_groups,
            trainer_cfg["rpn_initial_lr"],
            momentum=trainer_cfg["rpn_sgd_momentum"],
            nesterov=trainer_cfg["sgd_nesterov"],
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
