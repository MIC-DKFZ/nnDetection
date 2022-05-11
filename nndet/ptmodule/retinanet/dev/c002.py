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
from typing import Optional, Type

from nndet.core.abstract import AbstractOneStageDetector
from nndet.core.boxes.matcher import ATSSMatcher, Matcher
from nndet.core.boxes.sampler import HardNegativeSamplerBatched, SamplerType
from nndet.core.retina import BaseRetinaNet
from nndet.nn.blocks.basic import AbstractBlock, StackedConvBlock2
from nndet.nn.conv import BaseConvNormAct, ConvGroupLReLU, ConvInstanceLReLU
from nndet.nn.decoder.base import BaseUFPN, UFPNModular
from nndet.nn.encoder.abstract import AbstractEncoder
from nndet.nn.encoder.modular import Encoder
from nndet.nn.heads.classifier import BCECLassifier, FocalClassifier
from nndet.nn.heads.classifier.dense import DenseClassifier
from nndet.nn.heads.comb import BoxHeadAll, BoxHeadHNM
from nndet.nn.heads.comb.base import AnchorHead
from nndet.nn.heads.regressor import L1Regressor
from nndet.nn.heads.regressor.dense import DenseRegressor
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.retinanet.rn001 import RetinaNetModule
from nndet.utils.typing import CONVSEQ


@MODULE_REGISTRY.register
class RetinaNetC002(RetinaNetModule):
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

    # Not suported here; See `RetinaUNet`
    segmenter_cls = None


@MODULE_REGISTRY.register
class RetinaNetC002Focal(RetinaNetC002):
    """
    Focal Loss based RetinaNet
    """

    head_cls: Type[AnchorHead] = BoxHeadAll
    head_classifier_cls: Type[DenseClassifier] = FocalClassifier
    head_sampler_cls: Type[DenseRegressor] = None
