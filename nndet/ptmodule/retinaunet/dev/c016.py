# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import Type

from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.backbone.blueprints.conv import ConvBackbone
from nndet.nn.backbone.blueprints.nextconv import ConvNeXtBackbone
from nndet.nn.backbone.blueprints.resconv import ResConvBackbone
from nndet.nn.heads.classifier import FocalClassifier
from nndet.nn.heads.classifier.dense import BCECLassifier
from nndet.nn.heads.comb import BoxHeadAll, BoxHeadHNM
from nndet.nn.heads.comb.anchor_sampled import BoxHeadHNMV2
from nndet.nn.heads.comb.base import AnchorHead
from nndet.nn.heads.regressor import L1Regressor
from nndet.nn.layers.conv import ConvGroupLReLU, ConvInstanceLReLU
from nndet.nn.layers.initializer import InitHeV2
from nndet.nn.neck.abstract import AbstractNeck
from nndet.nn.neck.fpn import UFPN, UpFPN
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.retinaunet.run_v001 import RetinaUNetCV001Focal, RetinaUNetV001


@MODULE_REGISTRY.register
class RetinaUNetC016(RetinaUNetV001):
    backbone_cls: Type[AbstractBackbone] = ConvBackbone
    backbone_conv_cls = ConvInstanceLReLU

    neck_cls: Type[AbstractNeck] = UFPN
    neck_conv_cls = ConvInstanceLReLU

    head_cls = BoxHeadHNM
    head_conv_cls = ConvGroupLReLU
    head_regressor_cls = L1Regressor


@MODULE_REGISTRY.register
class RetinaUNetC016HeV2(RetinaUNetV001):
    backbone_conv_cls = partial(ConvInstanceLReLU, initializer=InitHeV2(mode="fan_out"))
    head_conv_cls = partial(ConvGroupLReLU, initializer=InitHeV2(mode="fan_out"))


@MODULE_REGISTRY.register
class RetinaUNetC016ResHeV2(RetinaUNetC016HeV2):
    backbone_cls: Type[AbstractBackbone] = ResConvBackbone


@MODULE_REGISTRY.register
class RetinaUNetC016Up(RetinaUNetC016):
    neck_cls: Type[AbstractNeck] = UpFPN


@MODULE_REGISTRY.register
class RetinaUNetC016Res(RetinaUNetC016):
    backbone_cls: Type[AbstractBackbone] = ResConvBackbone


@MODULE_REGISTRY.register
class RetinaUNetC016Focal(RetinaUNetCV001Focal):
    backbone_cls: Type[AbstractBackbone] = ConvBackbone
    backbone_conv_cls = ConvInstanceLReLU

    neck_cls: Type[AbstractNeck] = UFPN
    neck_conv_cls = ConvInstanceLReLU

    head_cls = BoxHeadAll
    head_conv_cls = ConvGroupLReLU
    head_sampler_cls = None
    head_regressor_cls = L1Regressor
    head_classifier_cls = FocalClassifier


@MODULE_REGISTRY.register
class RetinaUNetC016FocalNeXt(RetinaUNetC016Focal):
    backbone_cls: Type[AbstractBackbone] = ConvNeXtBackbone


class RetinaUNetC016V2(RetinaUNetC016):
    backbone_cls: Type[AbstractBackbone] = ConvBackbone
    backbone_conv_cls = ConvInstanceLReLU

    neck_cls: Type[AbstractNeck] = UFPN
    neck_conv_cls = ConvInstanceLReLU

    head_cls = BoxHeadHNM
    head_conv_cls = ConvGroupLReLU
    head_regressor_cls = L1Regressor

    head_classifier_cls = BCECLassifier
    head_cls: Type[AnchorHead] = BoxHeadHNMV2  # define class for head
