# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Type

from nndet.arch.blocks.basic import AbstractBlock, StackedConvBlock2
from nndet.arch.conv import BaseConvNormAct, ConvGroupRelu, ConvInstanceRelu
from nndet.arch.decoder.base import BaseUFPN, UFPNModular
from nndet.arch.encoder.abstract import AbstractEncoder
from nndet.arch.encoder.modular import Encoder
from nndet.arch.heads.classifier import CEClassifier
from nndet.arch.heads.classifier.dense import DenseClassifier
from nndet.arch.heads.comb import BoxHeadHNM
from nndet.arch.heads.comb.base import AnchorHead
from nndet.arch.heads.regressor import L1Regressor
from nndet.arch.heads.regressor.dense import DenseRegressor
from nndet.arch.heads.segmenter import DiCESegmenter, Segmenter
from nndet.core.abstract import AbstractOneStageDetector
from nndet.core.boxes.matcher import IoUMatcher, Matcher
from nndet.core.boxes.sampler import HardNegativeSamplerBatched, SamplerType
from nndet.core.retina import BaseRetinaNet
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

    backbone_cls: Type[AbstractEncoder] = ...  # define class for backbone
    backbone_conv_cls: Type[CONVSEQ] = ...  # conv class used for backbone
    backbone_block: Type[
        AbstractBlock
    ] = ...  # define central building block of backbone

    neck_cls: Type[BaseUFPN] = ...  # define class for neck
    neck_conv_cls: Type[CONVSEQ] = ...  # conv class used for neck

    head_cls: Type[AnchorHead] = ...  # define class for head
    head_conv_cls: Type[CONVSEQ] = ...  # conv class used for head
    head_classifier_cls: Type[DenseClassifier] = ...  # define class for head classifier
    head_regressor_cls: Type[DenseRegressor] = ...  # define class for head regressor
    # [optional] sampler class for negative mining
    # if None: no sampler will be given to the head
    head_sampler_cls: Optional[Type[SamplerType]] = ...

    matcher_cls: Type[Matcher] = ...  # define class to match anchors to ground truth
    segmenter_cls: Type[
        Segmenter
    ] = ...  # [optional] segmentation head as in RetinaUNet


class RetinaUNetBase(RetinaUNetModule):
    # Used for backwards compatibility
    # define detector cls
    detector_cls: Type[AbstractOneStageDetector] = BaseRetinaNet

    backbone_cls: Type[AbstractEncoder] = Encoder  # define class for backbone
    backbone_conv_cls: Type[
        BaseConvNormAct
    ] = ConvInstanceRelu  # conv class used for backbone
    backbone_block: Type[
        AbstractBlock
    ] = StackedConvBlock2  # define central building block of backbone

    neck_cls: Type[BaseUFPN] = UFPNModular  # define class for neck
    neck_conv_cls: Type[CONVSEQ] = ConvInstanceRelu  # conv class used for neck

    head_cls: Type[AnchorHead] = BoxHeadHNM  # define class for head
    head_conv_cls: Type[CONVSEQ] = ConvGroupRelu  # conv class used for head
    head_classifier_cls: Type[
        DenseClassifier
    ] = CEClassifier  # define class for head classifier
    head_regressor_cls: Type[
        DenseRegressor
    ] = L1Regressor  # define class for head regressor
    # [optional] sampler class for negative mining
    # if None: no sampler will be given to the head
    head_sampler_cls: Optional[Type[SamplerType]] = HardNegativeSamplerBatched

    matcher_cls: Type[
        Matcher
    ] = IoUMatcher  # define class to match anchors to ground truth
    segmenter_cls: Type[
        Segmenter
    ] = DiCESegmenter  # [optional] segmentation head as in RetinaUNet
