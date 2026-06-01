# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from nndet.nn.heads.classifier import FocalClassifier
from nndet.nn.heads.comb import BoxHeadAll, BoxHeadHNM
from nndet.nn.heads.regressor import L1Regressor
from nndet.nn.heads.segmenter import DiceTopKSegmenterFgBg, DiCETopKSegmenterFgBg
from nndet.nn.layers.conv import ConvBatchLReLU, ConvGroupLReLU, ConvInstanceLReLU
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.retinaunet.run_v001 import RetinaUNetCV001Focal, RetinaUNetV001


@MODULE_REGISTRY.register
class RetinaUNetC014(RetinaUNetV001):
    backbone_conv_cls = ConvInstanceLReLU
    neck_conv_cls = ConvInstanceLReLU
    head_conv_cls = ConvGroupLReLU

    head_cls = BoxHeadHNM
    head_regressor_cls = L1Regressor


@MODULE_REGISTRY.register
class RetinaUNetC014DiceTopK(RetinaUNetV001):
    segmenter_cls = DiceTopKSegmenterFgBg


@MODULE_REGISTRY.register
class RetinaUNetC014DiCETopK(RetinaUNetV001):
    segmenter_cls = DiCETopKSegmenterFgBg


@MODULE_REGISTRY.register
class RetinaUNetC014BN(RetinaUNetC014):
    backbone_conv_cls = ConvBatchLReLU
    neck_conv_cls = ConvBatchLReLU


@MODULE_REGISTRY.register
class RetinaUNetC014Focal(RetinaUNetCV001Focal):
    backbone_conv_cls = ConvInstanceLReLU
    neck_conv_cls = ConvInstanceLReLU
    head_conv_cls = ConvGroupLReLU

    head_cls = BoxHeadAll
    head_sampler_cls = None
    head_regressor_cls = L1Regressor
    head_classifier_cls = FocalClassifier


@MODULE_REGISTRY.register
class RetinaUNetC014BNFocal(RetinaUNetC014Focal):
    backbone_conv_cls = ConvBatchLReLU
    neck_conv_cls = ConvBatchLReLU
