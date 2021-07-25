from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001, RetinaUNetCV001Focal

from nndet.arch.heads.comb import (
    BoxHeadAll,
    BoxHeadHNM,
)
from nndet.arch.heads.classifier import (
    FocalClassifier,
)
from nndet.arch.heads.regressor import (
    L1Regressor
)
from nndet.arch.conv import (
    ConvInstanceLReLU,
    ConvGroupLReLU,
)


@MODULE_REGISTRY.register
class RetinaUNetC012(RetinaUNetV001):
    base_conv_cls = ConvInstanceLReLU
    head_conv_cls = ConvGroupLReLU

    head_cls = BoxHeadHNM
    head_regressor_cls = L1Regressor


@MODULE_REGISTRY.register
class RetinaUNetC012Focal(RetinaUNetCV001Focal):
    base_conv_cls = ConvInstanceLReLU
    head_conv_cls = ConvGroupLReLU

    head_cls = BoxHeadAll
    head_regressor_cls = L1Regressor
