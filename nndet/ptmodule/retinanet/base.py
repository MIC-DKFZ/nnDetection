"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import torch
from loguru import logger

from nndet.arch.blocks.basic import StackedConvBlock2
from nndet.arch.conv import ConvGroupRelu, ConvInstanceRelu
from nndet.arch.decoder.base import UFPNModular
from nndet.arch.encoder.modular import Encoder
from nndet.arch.heads.classifier import CEClassifier
from nndet.arch.heads.comb import BoxHeadHNM
from nndet.arch.heads.regressor import L1Regressor
from nndet.core.boxes.matcher import IoUMatcher
from nndet.core.boxes.sampler import HardNegativeSamplerBatched
from nndet.core.retina import BaseRetinaNet
from nndet.ptmodule.mixins.mode import BoxMixin
from nndet.ptmodule.mixins.model import SingleStageMixin
from nndet.ptmodule.mixins.prediction import BoxPredictionMixin
from nndet.ptmodule.module import LightningBaseModule
from nndet.training.learning_rate import LinearWarmupPolyLR
from nndet.training.optimizer import get_params_no_wd_on_norm


class RetinaNetModule(
    LightningBaseModule, BoxMixin, SingleStageMixin, BoxPredictionMixin
):
    # define detector cls
    detector_cls = BaseRetinaNet

    backbone_cls = Encoder  # define class for backbone
    backbone_conv_cls = ConvInstanceRelu  # conv class used for backbone
    backbone_block = StackedConvBlock2  # define central building block of backbone

    neck_cls = UFPNModular  # define class for neck
    neck_conv_cls = ConvInstanceRelu  # conv class used for neck

    head_cls = BoxHeadHNM  # define class for head
    head_conv_cls = ConvGroupRelu  # conv class used for head
    head_classifier_cls = CEClassifier  # define class for head classifier
    head_regressor_cls = L1Regressor  # define class for head regressor
    # [optional] sampler class for negative mining
    # if None: no sampler will be given to the head
    head_sampler_cls = HardNegativeSamplerBatched

    matcher_cls = IoUMatcher  # define class to match anchors to ground truth
    segmenter_cls = None  # [optional] segmentation head as in RetinaUNet

    def __init__(self, model_cfg: dict, trainer_cfg: dict, plan: dict, **kwargs):
        """
        RetinaUNet Lightning Module Skeleton

        Args:
            model_cfg: model configuration. Check :method:`from_config_plan`
                for more information
            trainer_cfg: trainer information
            plan: contains parameters which were derived from the planning
                stage
        """
        super().__init__(
            model_cfg=model_cfg, trainer_cfg=trainer_cfg, plan=plan, kwargs=kwargs
        )

    def configure_optimizers(self):
        """
        Configure optimizer and scheduler
        Base configuration is SGD with LinearWarmup and PolyLR learning rate
        schedule
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
