from nndet.arch.conv import (
    ConvBatchLReLU,
    ConvGroupLReLU,
    ConvGroupMish,
    ConvInstanceLReLU,
    ConvInstanceMish,
)
from nndet.arch.heads.classifier import FocalClassifier
from nndet.arch.heads.comb import BoxHeadAll, BoxHeadHNM
from nndet.arch.heads.regressor import L1Regressor
from nndet.arch.heads.segmenter import DiceTopKSegmenterFgBg, DiCETopKSegmenterFgBg
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.retinaunet.runv001 import RetinaUNetCV001Focal, RetinaUNetV001

"""
Bump version due to other changes
"""


@MODULE_REGISTRY.register
class RetinaUNetC015(RetinaUNetV001):
    backbone_conv_cls = ConvInstanceLReLU
    neck_conv_cls = ConvInstanceLReLU
    head_conv_cls = ConvGroupLReLU

    head_cls = BoxHeadHNM
    head_regressor_cls = L1Regressor


@MODULE_REGISTRY.register
class RetinaUNetC015MishHead(RetinaUNetC015):
    head_conv_cls = ConvGroupMish


@MODULE_REGISTRY.register
class RetinaUNetC015MishAll(RetinaUNetC015):
    backbone_conv_cls = ConvInstanceMish
    neck_conv_cls = ConvInstanceMish
    head_conv_cls = ConvGroupMish


@MODULE_REGISTRY.register
class RetinaUNetC015DiceTopK(RetinaUNetV001):
    segmenter_cls = DiceTopKSegmenterFgBg


@MODULE_REGISTRY.register
class RetinaUNetC015DiCETopK(RetinaUNetV001):
    segmenter_cls = DiCETopKSegmenterFgBg


@MODULE_REGISTRY.register
class RetinaUNetC015BN(RetinaUNetC015):
    backbone_conv_cls = ConvBatchLReLU
    neck_conv_cls = ConvBatchLReLU


@MODULE_REGISTRY.register
class RetinaUNetC015Focal(RetinaUNetCV001Focal):
    backbone_conv_cls = ConvInstanceLReLU
    neck_conv_cls = ConvInstanceLReLU
    head_conv_cls = ConvGroupLReLU

    head_cls = BoxHeadAll
    head_sampler_cls = None
    head_regressor_cls = L1Regressor
    head_classifier_cls = FocalClassifier


@MODULE_REGISTRY.register
class RetinaUNetC015BNFocal(RetinaUNetC015Focal):
    backbone_conv_cls = ConvBatchLReLU
    neck_conv_cls = ConvBatchLReLU
