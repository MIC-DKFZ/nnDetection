# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

import nndet
from nndet.ptmodule.optimizer import OPTIMIZER_REGISTRY
from nndet.training.learning_rate import (
    LinearWarmup,
    LinearWarmupPolyLR,
)
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


@OPTIMIZER_REGISTRY.register
class AdamWLWPoly_warmup:
    """
    AdamW Optimizer Mixin
    """

    @classmethod
    def configure_optimizers(
        cls,
        module: "nndet.ptmodule.module.LightningBaseModule",
        module_parts_to_train
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
            f"Running: initial_lr {trainer_cfg['initial_lr_warmup']} " f"weight_decay {trainer_cfg['weight_decay']} " f"AdamW"
        )

        wd_groups_backbone=[]
        wd_groups_head=[]
        wd_groups_channel_mapper=[]
        wd_groups_pos_embed=[]
        wd_groups_transformer=[]

        if "backbone" in module_parts_to_train:
            wd_groups_backbone = get_params_no_wd_on_norm(module.model.backbone, weight_decay=trainer_cfg["weight_decay"], on_conflict="skip")

        if "channel_mapper" in module_parts_to_train:
            wd_groups_channel_mapper = get_params_no_wd_on_norm(module.model.channel_mapper, weight_decay=trainer_cfg["weight_decay"], on_conflict="skip")

        if "pos_embed" in module_parts_to_train:
            wd_groups_pos_embed = get_params_no_wd_on_norm(module.model.pos_embed, weight_decay=trainer_cfg["weight_decay"], on_conflict="skip")

        if "transformer" in module_parts_to_train:
            wd_groups_transformer = get_params_no_wd_on_norm(module.model.transformer, weight_decay=trainer_cfg["weight_decay"], on_conflict="skip")

        if "head" in module_parts_to_train:
            wd_groups_head = get_params_no_wd_on_norm(module.model.head, weight_decay=trainer_cfg["weight_decay"], on_conflict="skip")

        combined_wd_groups = wd_groups_backbone+wd_groups_channel_mapper+ wd_groups_pos_embed+wd_groups_transformer+wd_groups_head

        dict_no_wd= {'params': [], 'weight_decay': 0.0}
        dict_wd = {'params': [], 'weight_decay': None}

        for d in combined_wd_groups:
            if d['weight_decay']==0.0:
                dict_no_wd['params'].extend(d['params'])
            else:
                dict_wd['params'].extend(d['params'])
                dict_wd['weight_decay']=d['weight_decay']

        combined_dict = [dict_wd, dict_no_wd]

        betas = (trainer_cfg["beta1"], trainer_cfg["beta2"])
        optimizer = torch.optim.AdamW(
            combined_dict,
            trainer_cfg["initial_lr_warmup"],
            weight_decay=trainer_cfg["weight_decay"],
            betas=betas,
            eps=trainer_cfg["eps"],
            amsgrad=trainer_cfg["amsgrad"],
        )
        num_iterations =  trainer_cfg['num_warmup_epochs']*trainer_cfg['num_train_batches_per_epoch']
        scheduler = LinearWarmup(
            optimizer=optimizer,
            warm_iterations=trainer_cfg["iterations_warmup"],
            warm_lr=trainer_cfg["warm_lr"],
            poly_gamma=trainer_cfg["poly_gamma"],
            num_iterations=num_iterations,
        )
        return {"optimizer":optimizer, "lr_scheduler": scheduler}

@OPTIMIZER_REGISTRY.register
class AdamWLWPoly_finetuning:
    """
    AdamW Optimizer Mixin
    """

    @classmethod
    def configure_optimizers(
        cls,
        module: "nndet.ptmodule.module.LightningBaseModule",
        module_parts_to_train
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
            f"Running: initial_lr {trainer_cfg['initial_lr_finetuning']} " f"weight_decay {trainer_cfg['weight_decay']} " f"AdamW"
        )

        wd_groups_backbone=[]
        wd_groups_head=[]
        wd_groups_channel_mapper=[]
        wd_groups_pos_embed=[]
        wd_groups_transformer=[]

        if "backbone" in module_parts_to_train:
            wd_groups_backbone = get_params_no_wd_on_norm(module.model.backbone, weight_decay=trainer_cfg["weight_decay"], on_conflict="skip")

        if "channel_mapper" in module_parts_to_train:
            wd_groups_channel_mapper = get_params_no_wd_on_norm(module.model.channel_mapper, weight_decay=trainer_cfg["weight_decay"], on_conflict="skip")

        if "pos_embed" in module_parts_to_train:
            wd_groups_pos_embed = get_params_no_wd_on_norm(module.model.pos_embed, weight_decay=trainer_cfg["weight_decay"], on_conflict="skip")

        if "transformer" in module_parts_to_train:
            wd_groups_transformer = get_params_no_wd_on_norm(module.model.transformer, weight_decay=trainer_cfg["weight_decay"], on_conflict="skip")

        if "head" in module_parts_to_train:
            wd_groups_head = get_params_no_wd_on_norm(module.model.head, weight_decay=trainer_cfg["weight_decay"], on_conflict="skip")

        combined_wd_groups = wd_groups_backbone+wd_groups_channel_mapper+ wd_groups_pos_embed+wd_groups_transformer+wd_groups_head

        dict_no_wd= {'params': [], 'weight_decay': 0.0}
        dict_wd = {'params': [], 'weight_decay': None}

        for d in combined_wd_groups:
            if d['weight_decay']==0.0:
                dict_no_wd['params'].extend(d['params'])
            else:
                dict_wd['params'].extend(d['params'])
                dict_wd['weight_decay']=d['weight_decay']

        combined_dict = [dict_wd, dict_no_wd]

        betas = (trainer_cfg["beta1"], trainer_cfg["beta2"])
        optimizer = torch.optim.AdamW(
            combined_dict,
            trainer_cfg["initial_lr_finetuning"],
            weight_decay=trainer_cfg["weight_decay"],
            betas=betas,
            eps=trainer_cfg["eps"],
            amsgrad=trainer_cfg["amsgrad"],
        )

        num_iterations = trainer_cfg['max_num_epochs']*trainer_cfg['num_train_batches_per_epoch']
        scheduler = LinearWarmupPolyLR(
            optimizer=optimizer,
            warm_iterations=trainer_cfg["warm_iterations_finetuning"],
            warm_lr=trainer_cfg["warm_lr_finetuning"],
            poly_gamma=trainer_cfg["poly_gamma"],
            num_iterations=num_iterations
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
