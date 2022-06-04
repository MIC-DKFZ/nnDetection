from typing import Type

from nndet.arch.conv import ConvGroupLReLU, ConvInstanceLReLU, ConvInstanceMish
from nndet.arch.decoder.base import UPAN, BaseUFPN
from nndet.arch.heads.classifier import FocalClassifier
from nndet.arch.heads.comb import BoxHeadAll, BoxHeadHNM
from nndet.arch.heads.regressor import L1Regressor
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.retinaunet.runv001 import RetinaUNetCV001Focal, RetinaUNetV001
from nndet.utils.typing import CONVSEQ


@MODULE_REGISTRY.register
class RetinaUNetC016(RetinaUNetV001):
    backbone_conv_cls = ConvInstanceLReLU
    neck_conv_cls = ConvInstanceLReLU
    head_conv_cls = ConvGroupLReLU

    head_cls = BoxHeadHNM
    head_regressor_cls = L1Regressor


@MODULE_REGISTRY.register
class RetinaUNetC016Focal(RetinaUNetCV001Focal):
    backbone_conv_cls = ConvInstanceLReLU
    neck_conv_cls = ConvInstanceLReLU
    head_conv_cls = ConvGroupLReLU

    head_cls = BoxHeadAll
    head_sampler_cls = None
    head_regressor_cls = L1Regressor
    head_classifier_cls = FocalClassifier


@MODULE_REGISTRY.register
class RetinaUNetC016PAN(RetinaUNetC016):
    neck_conv_cls: Type[CONVSEQ] = ConvGroupLReLU
    neck_cls: Type[BaseUFPN] = UPAN


@MODULE_REGISTRY.register
class RetinaUNetC016PANInstMish(RetinaUNetC016):
    neck_conv_cls: Type[CONVSEQ] = ConvInstanceMish
    neck_cls: Type[BaseUFPN] = UPAN


@MODULE_REGISTRY.register
class RetinaUNetC016PANGroupMish(RetinaUNetC016):
    neck_conv_cls: Type[CONVSEQ] = ConvInstanceMish
    neck_cls: Type[BaseUFPN] = UPAN
