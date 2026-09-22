# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import torch

from nndet.ptmodule.optimizer import OPTIMIZER_REGISTRY
from nndet.training.callbacks import LossNaNError


class TwoPhaseWarmupMixin:
    """
    Two-phase warmup finetuning: freeze most of the model for
    trainer_cfg.num_warmup_epochs, training only phase1_parts_to_train
    with trainer_cfg.opt_class_1, then switch to training
    phase2_parts_to_train with trainer_cfg.opt_class_2 for the rest of
    training.

    Concrete warmup classes only need to set phase1_parts_to_train and
    phase2_parts_to_train (the component names accepted by the model's
    own optimizer-parts split, e.g. "backbone"/"neck"/"head" for
    RetinaUNet-style models or "backbone"/"channel_mapper"/"transformer"/
    "head" for DETR-style models) -- everything else here is identical
    regardless of backbone/head family.
    """

    automatic_optimization = False

    #: parts trained during trainer_cfg.num_warmup_epochs, with opt_class_1
    phase1_parts_to_train = None
    #: parts trained after the warmup phase, with opt_class_2
    phase2_parts_to_train = None

    def configure_optimizers(self):
        optimizer_1 = OPTIMIZER_REGISTRY[self.trainer_cfg["opt_class_1"]].configure_optimizers(
            self, self.phase1_parts_to_train
        )
        optimizer_2 = OPTIMIZER_REGISTRY[self.trainer_cfg["opt_class_2"]].configure_optimizers(
            self, self.phase2_parts_to_train
        )
        return (
            [optimizer_1["optimizer"], optimizer_2["optimizer"]],
            [optimizer_1["lr_scheduler"], optimizer_2["lr_scheduler"]],
        )

    def training_step(self, batch, batch_idx):
        """
        Computes a single training step
        See :class:`BaseRetinaNet` for more information
        """
        with torch.no_grad():
            batch = self.pre_trafo(**batch)

        if "target" in batch:  # free memory from numbered instance seg
            del batch["target"]
        targets = {key: item for key, item in batch.items() if "target_" in key}

        if "target_seg" in targets:
            # [optional] add semantic segmentation to targets if available
            # Remove channel dimension of semantic segmentation
            targets["target_seg"] = targets["target_seg"][:, 0]
        if "target_binary_masks" in targets:
            # [optional] add binary masks to targets if available
            targets["target_binary_masks"] = targets["target_binary_masks"]

        if self.do_channels_last:
            if self.dim == 3:
                _data = batch["data"].to(memory_format=torch.channels_last_3d)
            else:
                _data = batch["data"].to(memory_format=torch.channels_last)
        else:
            _data = batch["data"]

        if self.current_epoch < self.trainer_cfg["num_warmup_epochs"]:
            optimizer = self.optimizers()[0]
            scheduler = self.lr_schedulers()[0]
        else:
            optimizer = self.optimizers()[1]
            scheduler = self.lr_schedulers()[1]

        optimizer.zero_grad()

        losses = self.model.train_step(
            images=_data,
            targets=targets,
            batch_num=batch_idx,
        )
        info = {key: losses.pop(key) for key in list(losses.keys()) if key.startswith("__")}
        loss = sum(losses.values())

        self.manual_backward(loss)

        optimizer.step()
        scheduler.step()

        if torch.isnan(loss):
            raise LossNaNError("Found NaN loss in training step.")

        out = {"loss": loss.detach().item(), **{f"loss_{key}": l.detach().item() for key, l in losses.items()}, **info}
        self.log("train_step_loss", out["loss"], prog_bar=True, logger=False, batch_size=1)
        self.training_step_outputs.append(out)
        return loss
