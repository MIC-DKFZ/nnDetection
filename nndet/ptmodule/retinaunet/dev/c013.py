# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from nndet.core.boxes.matcher import IoUMatcher
from nndet.nn.heads.classifier import FocalClassifier
from nndet.nn.heads.comb import BoxHeadAll, BoxHeadHNM
from nndet.nn.heads.regressor import L1Regressor
from nndet.nn.layers.conv import ConvBatchLReLU, ConvGroupLReLU, ConvInstanceLReLU
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.retinaunet.run_v001 import RetinaUNetCV001Focal, RetinaUNetV001


@MODULE_REGISTRY.register
class RetinaUNetC013(RetinaUNetV001):
    backbone_conv_cls = ConvInstanceLReLU
    neck_conv_cls = ConvInstanceLReLU
    head_conv_cls = ConvGroupLReLU

    head_cls = BoxHeadHNM
    head_regressor_cls = L1Regressor


@MODULE_REGISTRY.register
class RetinaUNetC013BN(RetinaUNetC013):
    backbone_conv_cls = ConvBatchLReLU
    neck_conv_cls = ConvBatchLReLU


@MODULE_REGISTRY.register
class RetinaUNetC013IoU(RetinaUNetC013):
    matcher_cls = IoUMatcher  # define class to match anchors to ground truth


@MODULE_REGISTRY.register
class RetinaUNetC013IoUBN(RetinaUNetC013IoU):
    backbone_conv_cls = ConvBatchLReLU
    neck_conv_cls = ConvBatchLReLU


@MODULE_REGISTRY.register
class RetinaUNetC013Focal(RetinaUNetCV001Focal):
    backbone_conv_cls = ConvInstanceLReLU
    neck_conv_cls = ConvInstanceLReLU
    head_conv_cls = ConvGroupLReLU

    head_cls = BoxHeadAll
    head_sampler_cls = None
    head_regressor_cls = L1Regressor
    head_classifier_cls = FocalClassifier


@MODULE_REGISTRY.register
class RetinaUNetC013FocalBN(RetinaUNetC013Focal):
    backbone_conv_cls = ConvBatchLReLU
    neck_conv_cls = ConvBatchLReLU


@MODULE_REGISTRY.register
class RetinaUNetC013FocalIoU(RetinaUNetC013Focal):
    matcher_cls = IoUMatcher  # define class to match anchors to ground truth


@MODULE_REGISTRY.register
class RetinaUNetC013FocalIoUBN(RetinaUNetC013FocalIoU):
    backbone_conv_cls = ConvBatchLReLU
    neck_conv_cls = ConvBatchLReLU
