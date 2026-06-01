# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Type

from nndet.core.boxes.matcher import ATSSMatcher
from nndet.core.boxes.sampler import HardNegativeSamplerBatched
from nndet.core.post.box import BoxPostprocessing, CrossLevelBoxPostprocessing
from nndet.core.retina import BaseRetinaNet
from nndet.nn.backbone.blueprints.conv import ConvBackbone
from nndet.nn.heads.classifier import BCECLassifier, FocalClassifier
from nndet.nn.heads.comb import BoxHeadAll, BoxHeadHNMNative
from nndet.nn.heads.regressor import GIoURegressor
from nndet.nn.heads.segmenter import DiCESegmenterFgBg
from nndet.nn.layers.conv import ConvGroupRelu, ConvInstanceRelu
from nndet.nn.neck.fpn import UFPN
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.mixins.evaluation import BoxEvalMixin, SemanticFgEvalMixin
from nndet.ptmodule.mixins.model import SingleStageMixin
from nndet.ptmodule.mixins.prediction import BoxPredictionMixin
from nndet.ptmodule.mixins.prepare import BoxesPrepareMixin, SemanticFgPrepareMixin
from nndet.ptmodule.module import LightningBaseModule


@MODULE_REGISTRY.register
class RetinaUNetV001(
    LightningBaseModule,  # Detection Base
    # prepare inputs
    BoxesPrepareMixin,
    SemanticFgPrepareMixin,
    # evaluation
    BoxEvalMixin,
    SemanticFgEvalMixin,
    SingleStageMixin,  # Single Stage Detector
    # inference
    BoxPredictionMixin,  # Bounding Box Sweep
):
    # define detector cls
    detector_cls = BaseRetinaNet

    backbone_cls = ConvBackbone  # define class for backbone
    backbone_conv_cls = ConvInstanceRelu

    neck_cls = UFPN  # define class for neck
    neck_conv_cls = ConvInstanceRelu

    head_cls = BoxHeadHNMNative
    head_conv_cls = ConvGroupRelu
    head_classifier_cls = BCECLassifier
    head_regressor_cls = GIoURegressor
    # [optional] sampler class for negative mining
    # if None: no sampler will be given to the head
    head_sampler_cls = HardNegativeSamplerBatched

    matcher_cls = ATSSMatcher
    box_post_cls: Type[BoxPostprocessing] = CrossLevelBoxPostprocessing  # define box postprocessing strategy

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
