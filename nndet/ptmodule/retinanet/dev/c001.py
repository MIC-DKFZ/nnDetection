# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Type

from nndet.arch.blocks.basic import AbstractBlock, StackedConvBlock2, StackedConvBlock3
from nndet.arch.conv import BaseConvNormAct, ConvGroupLReLU, ConvInstanceLReLU
from nndet.arch.decoder.base import BaseUFPN, UFPNModular
from nndet.arch.encoder.abstract import AbstractEncoder
from nndet.arch.encoder.modular import Encoder
from nndet.arch.heads.classifier import BCECLassifier, FocalClassifier
from nndet.arch.heads.classifier.dense import DenseClassifier
from nndet.arch.heads.comb import BoxHeadAll, BoxHeadHNM
from nndet.arch.heads.comb.base import AnchorHead
from nndet.arch.heads.regressor import L1Regressor
from nndet.arch.heads.regressor.dense import DenseRegressor
from nndet.core.abstract import AbstractOneStageDetector
from nndet.core.boxes.matcher import ATSSMatcher, Matcher
from nndet.core.boxes.sampler import HardNegativeSamplerBatched, SamplerType
from nndet.core.retina import BaseRetinaNet
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.retinanet.rn001 import RetinaNetModule
from nndet.utils.typing import CONVSEQ


@MODULE_REGISTRY.register
class RetinaNetC001(RetinaNetModule):
    # define detector cls
    detector_cls: Type[AbstractOneStageDetector] = BaseRetinaNet

    backbone_cls: Type[AbstractEncoder] = Encoder  # define class for backbone
    backbone_conv_cls: Type[
        BaseConvNormAct
    ] = ConvInstanceLReLU  # conv class used for backbone
    backbone_block: Type[
        AbstractBlock
    ] = StackedConvBlock2  # define central building block of backbone

    neck_cls: Type[BaseUFPN] = UFPNModular  # define class for neck
    neck_conv_cls: Type[CONVSEQ] = ConvInstanceLReLU  # conv class used for neck

    head_cls: Type[AnchorHead] = BoxHeadHNM  # define class for head
    head_conv_cls: Type[CONVSEQ] = ConvGroupLReLU  # conv class used for head
    head_classifier_cls: Type[
        DenseClassifier
    ] = BCECLassifier  # define class for head classifier
    head_regressor_cls: Type[
        DenseRegressor
    ] = L1Regressor  # define class for head regressor
    # [optional] sampler class for negative mining
    # if None: no sampler will be given to the head
    head_sampler_cls: Optional[Type[SamplerType]] = HardNegativeSamplerBatched

    matcher_cls: Type[
        Matcher
    ] = ATSSMatcher  # define class to match anchors to ground truth


@MODULE_REGISTRY.register
class RetinaNetC001Focal(RetinaNetModule):
    """
    Focal Loss based V001 RetinaNet
    (only intended for easy subclassing and not used in nnDetection V0.1)
    """

    head_cls: Type[AnchorHead] = BoxHeadAll
    head_classifier_cls: Type[DenseClassifier] = FocalClassifier
    head_sampler_cls: Type[DenseRegressor] = None


@MODULE_REGISTRY.register
class RetinaNetC001FocalC3(RetinaNetC001Focal):
    backbone_block: Type[
        AbstractBlock
    ] = StackedConvBlock3  # define central building block of backbone
