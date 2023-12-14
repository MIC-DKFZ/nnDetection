# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Type

from nndet.core.abstract import AbstractOneStageDetector
from nndet.core.boxes.matcher import IoUMatcher, Matcher
from nndet.core.boxes.sampler import AbstractSampler, HardNegativeSamplerBatched
from nndet.core.post.box import BoxPostprocessing, CrossLevelBoxPostprocessing
from nndet.core.retina import BaseRetinaNet
from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.backbone.blueprints.conv import ConvBackbone
from nndet.nn.heads.classifier import CEClassifier
from nndet.nn.heads.classifier.dense import DenseClassifier
from nndet.nn.heads.comb import BoxHeadHNM
from nndet.nn.heads.comb.base import AnchorHead
from nndet.nn.heads.regressor import L1Regressor
from nndet.nn.heads.regressor.dense import DenseRegressor
from nndet.nn.heads.segmenter import DiCESegmenter, Segmenter
from nndet.nn.layers.conv import BaseConvNormAct, ConvGroupRelu, ConvInstanceRelu
from nndet.nn.neck.abstract import AbstractNeck
from nndet.nn.neck.fpn import UFPN
from nndet.ptmodule.mixins.evaluation import BoxEvalMixin, SemanticEvalMixin
from nndet.ptmodule.mixins.model import SingleStageMixin
from nndet.ptmodule.mixins.prediction import BoxPredictionMixin
from nndet.ptmodule.mixins.prepare import BoxesPrepareMixin, SemanticPrepareMixin
from nndet.ptmodule.module import LightningBaseModule
from nndet.utils.typing import CONVSEQ


class RetinaUNetModule(
    LightningBaseModule,  # Detection Base
    SemanticPrepareMixin,  # prepare batch for semantic segmentation training
    BoxesPrepareMixin,  # prepare batch for box training
    SemanticEvalMixin,  # Semantic Segmentation Evaluation
    BoxEvalMixin,  # Boundig Box Evaluation
    SingleStageMixin,  # Single Stage Detector
    BoxPredictionMixin,  # Bounding Box Sweep
):
    # define detector cls
    detector_cls: Type[AbstractOneStageDetector] = BaseRetinaNet

    backbone_cls: Type[AbstractBackbone] = ...  # define class for backbone
    backbone_conv_cls: Type[CONVSEQ] = ...  # conv class used for backbone

    neck_cls: Type[AbstractNeck] = ...  # define class for neck
    neck_conv_cls: Type[CONVSEQ] = ...  # conv class used for neck

    head_cls: Type[AnchorHead] = ...  # define class for head
    head_conv_cls: Type[CONVSEQ] = ...  # conv class used for head
    head_classifier_cls: Type[DenseClassifier] = ...  # define class for head classifier
    head_regressor_cls: Type[DenseRegressor] = ...  # define class for head regressor
    # [optional] sampler class for negative mining
    # if None: no sampler will be given to the head
    head_sampler_cls: Optional[Type[AbstractSampler]] = ...

    matcher_cls: Type[Matcher] = ...  # define class to match anchors to ground truth
    box_post_cls: Type[BoxPostprocessing] = ...  # define box postprocessing strategy

    segmenter_cls: Type[Segmenter] = ...  # [optional] segmentation head as in RetinaUNet


class RetinaUNetBase(RetinaUNetModule):
    # Used for backwards compatibility
    # define detector cls
    detector_cls: Type[AbstractOneStageDetector] = BaseRetinaNet

    backbone_cls: Type[AbstractBackbone] = ConvBackbone  # define class for backbone
    backbone_conv_cls: Type[BaseConvNormAct] = ConvInstanceRelu  # conv class used for backbone

    neck_cls: Type[AbstractNeck] = UFPN  # define class for neck
    neck_conv_cls: Type[CONVSEQ] = ConvInstanceRelu  # conv class used for neck

    head_cls: Type[AnchorHead] = BoxHeadHNM  # define class for head
    head_conv_cls: Type[CONVSEQ] = ConvGroupRelu  # conv class used for head
    head_classifier_cls: Type[DenseClassifier] = CEClassifier  # define class for head classifier
    head_regressor_cls: Type[DenseRegressor] = L1Regressor  # define class for head regressor
    # [optional] sampler class for negative mining
    # if None: no sampler will be given to the head
    head_sampler_cls: Optional[Type[AbstractSampler]] = HardNegativeSamplerBatched

    matcher_cls: Type[Matcher] = IoUMatcher  # define class to match anchors to ground truth
    box_post_cls: Type[BoxPostprocessing] = CrossLevelBoxPostprocessing  # define box postprocessing strategy

    segmenter_cls: Type[Segmenter] = DiCESegmenter  # [optional] segmentation head as in RetinaUNet
