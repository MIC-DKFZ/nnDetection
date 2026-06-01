# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import partial
from typing import Optional, Type

from nndet.core.abstract import AbstractOneStageDetector
from nndet.core.boxes.matcher import ATSSMatcher
from nndet.core.boxes.matcher.base import Matcher
from nndet.core.boxes.sampler import AbstractSampler, HardNegativeSamplerBatched
from nndet.core.post.box import BoxPostprocessing, CrossLevelBoxPostprocessing
from nndet.core.retina import BaseRetinaNet
from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.backbone.blueprints.resconv import ConvBackbone, ResConvBackbone
from nndet.nn.heads.classifier import BCECLassifier, FocalClassifier
from nndet.nn.heads.classifier.dense import DenseClassifier
from nndet.nn.heads.comb import BoxHeadAll
from nndet.nn.heads.comb.anchor_sampled import BoxHeadHNM, BoxHeadHNMV2
from nndet.nn.heads.comb.base import AnchorHead
from nndet.nn.heads.regressor.dense import DenseRegressor, L1Regressor
from nndet.nn.heads.segmenter import DiCESegmenterFgBg, Segmenter
from nndet.nn.layers.conv.group import ConvGroupLReLU, ConvGroupMish
from nndet.nn.layers.conv.instance import ConvInstanceLReLU
from nndet.nn.layers.initializer import InitHeV2
from nndet.nn.neck.abstract import AbstractNeck
from nndet.nn.neck.fpn import UFPN
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.mixins.evaluation import BoxEvalMixin, SemanticFgEvalMixin
from nndet.ptmodule.mixins.model import SingleStageMixin
from nndet.ptmodule.mixins.prediction import BoxPredictionMixin
from nndet.ptmodule.mixins.prepare import BoxesPrepareMixin, SemanticFgPrepareMixin
from nndet.ptmodule.module import LightningBaseModule
from nndet.utils.typing import CONVSEQ


@MODULE_REGISTRY.register
class RetinaUNetHNMC017(
    LightningBaseModule,  # Detection Base
    SemanticFgPrepareMixin,  # prepare batch for semantic segmentation training
    BoxesPrepareMixin,  # prepare batch for box training
    SemanticFgEvalMixin,  # Semantic Segmentation Evaluation
    BoxEvalMixin,  # Boundig Box Evaluation
    SingleStageMixin,  # Single Stage Detector
    BoxPredictionMixin,  # Bounding Box Sweep
):
    # define detector cls
    detector_cls: Type[AbstractOneStageDetector] = BaseRetinaNet

    backbone_cls: Type[AbstractBackbone] = ConvBackbone  # define class for backbone
    backbone_conv_cls: Type[CONVSEQ] = partial(
        ConvInstanceLReLU, initializer=InitHeV2(mode="fan_out")
    )  # conv class used for backbone

    neck_cls: Type[AbstractNeck] = UFPN  # define class for neck
    neck_conv_cls: Type[CONVSEQ] = partial(
        ConvGroupLReLU, initializer=InitHeV2(mode="fan_out")
    )  # conv class used for neck

    head_cls: Type[AnchorHead] = BoxHeadHNMV2  # define class for head
    head_conv_cls: Type[CONVSEQ] = ConvGroupLReLU  # conv class used for head
    head_classifier_cls: Type[DenseClassifier] = BCECLassifier  # define class for head classifier
    head_regressor_cls: Type[DenseRegressor] = L1Regressor  # define class for head regressor
    # [optional] sampler class for negative mining
    # if None: no sampler will be given to the head
    head_sampler_cls: Optional[Type[AbstractSampler]] = HardNegativeSamplerBatched

    matcher_cls: Type[Matcher] = ATSSMatcher  # define class to match anchors to ground truth
    box_post_cls: Type[BoxPostprocessing] = CrossLevelBoxPostprocessing  # define box postprocessing strategy

    segmenter_cls: Type[Segmenter] = DiCESegmenterFgBg  # [optional] segmentation head as in RetinaUNet


@MODULE_REGISTRY.register
class RetinaUNetHNMC017MH(RetinaUNetHNMC017):
    """
    Use old head since V2 Head produced worse results
    """

    head_cls: Type[AnchorHead] = BoxHeadHNM  # define class for head


@MODULE_REGISTRY.register
class RetinaUNetFocalC017(RetinaUNetHNMC017):
    """
    Focal Loss based RetinaUNet V002
    """

    head_cls: Type[AnchorHead] = BoxHeadAll  # define class for head
    head_classifier_cls: Type[DenseClassifier] = FocalClassifier  # define class for head classifier
    # [optional] sampler class for negative mining
    head_sampler_cls: Optional[Type[AbstractSampler]] = None


@MODULE_REGISTRY.register
class RetinaUNetFocalC017Mish(RetinaUNetFocalC017):
    """
    Focal Loss based RetinaUNet V002
    """

    head_conv_cls: Type[CONVSEQ] = ConvGroupMish


@MODULE_REGISTRY.register
class RetinaUNetFocalC017Res(RetinaUNetFocalC017):
    """
    Residual Conv Backbone
    """

    backbone_cls: Type[AbstractBackbone] = ResConvBackbone  # define class for backbone
