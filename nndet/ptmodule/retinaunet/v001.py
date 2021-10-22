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

from nndet.arch.blocks.basic import StackedConvBlock2
from nndet.arch.conv import ConvGroupRelu, ConvInstanceRelu
from nndet.arch.decoder.base import UFPNModular
from nndet.arch.encoder.modular import Encoder
from nndet.arch.heads.classifier import BCECLassifier, FocalClassifier
from nndet.arch.heads.comb import BoxHeadAll, BoxHeadHNMNative
from nndet.arch.heads.regressor import GIoURegressor
from nndet.arch.heads.segmenter import DiCESegmenterFgBg
from nndet.core.boxes.matcher import ATSSMatcher
from nndet.core.boxes.sampler import HardNegativeSamplerBatched
from nndet.core.retina import BaseRetinaNet
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.mixins.evaluation import BoxEvalMixin, SemanticFgEvalMixin
from nndet.ptmodule.mixins.model import SingleStageMixin
from nndet.ptmodule.mixins.optimizer import SGDDefaultMixin
from nndet.ptmodule.mixins.prediction import BoxPredictionMixin
from nndet.ptmodule.mixins.prepare import BoxPrepareMixin, SemanticFgPrepareMixin
from nndet.ptmodule.module import LightningBaseModule


@MODULE_REGISTRY.register
class RetinaUNetV001(
    SGDDefaultMixin,  # Default SGD optimization
    LightningBaseModule,  # Detection Base
    # prepare inputs
    BoxPrepareMixin,
    SemanticFgPrepareMixin,
    # evaluation
    BoxEvalMixin,
    SemanticFgEvalMixin,
    SingleStageMixin,  # Single Stage Detector
    BoxPredictionMixin,  # Bounding Box Sweep
):
    # define detector cls
    detector_cls = BaseRetinaNet

    backbone_cls = Encoder  # define class for backbone
    backbone_conv_cls = ConvInstanceRelu
    backbone_block = StackedConvBlock2  # define central building block of backbone

    neck_cls = UFPNModular  # define class for neck
    neck_conv_cls = ConvInstanceRelu

    head_cls = BoxHeadHNMNative
    head_conv_cls = ConvGroupRelu
    head_classifier_cls = BCECLassifier
    head_regressor_cls = GIoURegressor
    # [optional] sampler class for negative mining
    # if None: no sampler will be given to the head
    head_sampler_cls = HardNegativeSamplerBatched

    matcher_cls = ATSSMatcher
    segmenter_cls = DiCESegmenterFgBg


@MODULE_REGISTRY.register
class RetinaUNetCV001Focal(RetinaUNetV001):
    """
    Focal Loss based V001 RetinaUNet
    (only intended for easy subclassing and not used in nnDetection V0.1)
    """

    head_cls = BoxHeadAll
    head_classifier_cls = FocalClassifier
    head_sampler_cls = None
