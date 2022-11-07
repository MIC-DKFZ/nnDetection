import torch
from loguru import logger

from nndet.ptmodule.optimizer import OPTIMIZER_REGISTRY
from nndet.training.learning_rate import LinearWarmupPolyLR
from nndet.training.optimizer import get_params_no_wd_on_norm


@OPTIMIZER_REGISTRY.register
class AdamWPoly:
    """
    AdamW Optimizer Mixin which supports a different learning rate for backbone and the rest for pretrained backbone
    support
    """

    @classmethod
    def configure_optimizers(cls, module):
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
            f"Running: initial_lr {trainer_cfg['initial_lr']} "
            f"weight_decay {trainer_cfg['weight_decay']} "
            f"AdamW with LinearWarmupPolyLR Scheduler"
        )
        # Obtain all parameters that are not in the backbone (no batch/layer norms outside the backbone) -> normal
        # learning rate and weight decay applies
        detr_params = {"params": [p for n, p in module.named_parameters() if "backbone" not in n and p.requires_grad]}

        # Differentiate between normal layers and normalization layers which shouldn't have weight decay
        if hasattr(module.model, "backbone"):
            param_groups = get_params_no_wd_on_norm(module.model.backbone, weight_decay=trainer_cfg["weight_decay"])
        elif hasattr(module.model.detr, "backbone"):
            param_groups = get_params_no_wd_on_norm(
                module.model.detr.backbone, weight_decay=trainer_cfg["weight_decay"]
            )
        else:
            raise "Module has no Backbone to assign smaller learning rate"

        lr_dict = {"lr": trainer_cfg["lr_backbone"]}
        for x in param_groups:
            x.update(lr_dict)
        param_groups.append(detr_params)

        betas = (trainer_cfg["beta1"], trainer_cfg["beta2"])
        optimizer = torch.optim.AdamW(
            param_groups,
            trainer_cfg["initial_lr"],
            weight_decay=trainer_cfg["weight_decay"],
            betas=betas,
            eps=trainer_cfg["eps"],
            amsgrad=trainer_cfg["amsgrad"],
        )

        # configure lr scheduler
        num_iterations = trainer_cfg["max_num_epochs"] * trainer_cfg["num_train_batches_per_epoch"]
        scheduler = LinearWarmupPolyLR(
            optimizer=optimizer,
            warm_iterations=trainer_cfg["warm_iterations"],
            warm_lr=trainer_cfg["warm_lr"],
            poly_gamma=trainer_cfg["poly_gamma"],
            num_iterations=num_iterations,
        )
        return [optimizer], {"scheduler": scheduler, "interval": "step"}
