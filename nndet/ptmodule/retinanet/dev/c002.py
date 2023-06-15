# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Type

from nndet.core.abstract import AbstractOneStageDetector
from nndet.core.boxes.matcher import ATSSMatcher, Matcher
from nndet.core.boxes.sampler import AbstractSampler, HardNegativeSamplerBatched
from nndet.core.post.box import BoxPostprocessing, CrossLevelBoxPostprocessing
from nndet.core.retina import BaseRetinaNet
from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.backbone.blueprints.conv import ConvBackbone
from nndet.nn.heads.classifier import BCECLassifier, FocalClassifier
from nndet.nn.heads.classifier.dense import DenseClassifier
from nndet.nn.heads.comb import BoxHeadAll, BoxHeadHNM
from nndet.nn.heads.comb.base import AnchorHead
from nndet.nn.heads.regressor import L1Regressor
from nndet.nn.heads.regressor.dense import DenseRegressor
from nndet.nn.layers.conv import BaseConvNormAct, ConvGroupLReLU, ConvInstanceLReLU
from nndet.nn.neck.abstract import AbstractNeck
from nndet.nn.neck.fpn import FPN
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.retinanet.rnv001 import RetinaNetModule
from nndet.utils.typing import CONVSEQ


@MODULE_REGISTRY.register
class RetinaNetC002(RetinaNetModule):
    # define detector cls
    detector_cls: Type[AbstractOneStageDetector] = BaseRetinaNet

    backbone_cls: Type[AbstractBackbone] = ConvBackbone  # define class for backbone
    backbone_conv_cls: Type[BaseConvNormAct] = ConvInstanceLReLU  # conv class used for backbone

    neck_cls: Type[AbstractNeck] = FPN  # define class for neck
    neck_conv_cls: Type[CONVSEQ] = ConvInstanceLReLU  # conv class used for neck

    head_cls: Type[AnchorHead] = BoxHeadHNM  # define class for head
    head_conv_cls: Type[CONVSEQ] = ConvGroupLReLU  # conv class used for head
    head_classifier_cls: Type[DenseClassifier] = BCECLassifier  # define class for head classifier
    head_regressor_cls: Type[DenseRegressor] = L1Regressor  # define class for head regressor
    # [optional] sampler class for negative mining
    # if None: no sampler will be given to the head
    head_sampler_cls: Optional[Type[AbstractSampler]] = HardNegativeSamplerBatched

    matcher_cls: Type[Matcher] = ATSSMatcher  # define class to match anchors to ground truth
    box_post_cls: Type[BoxPostprocessing] = CrossLevelBoxPostprocessing  # define box postprocessing strategy

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
