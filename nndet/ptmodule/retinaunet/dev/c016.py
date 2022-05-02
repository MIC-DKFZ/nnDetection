from typing import Type

from nndet.arch.conv import ConvGroupLReLU, ConvInstanceLReLU
from nndet.arch.heads.classifier import FocalClassifier
from nndet.arch.heads.comb import BoxHeadAll, BoxHeadHNM
from nndet.arch.heads.regressor import L1Regressor
from nndet.inference.ensembler.base import BaseEnsembler
from nndet.inference.ensembler.detection import BoxEnsemblerSelectiveV2
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.retinaunet.runv001 import RetinaUNetCV001Focal, RetinaUNetV001


@MODULE_REGISTRY.register
class RetinaUNetC016(RetinaUNetV001):
    backbone_conv_cls = ConvInstanceLReLU
    neck_conv_cls = ConvInstanceLReLU
    head_conv_cls = ConvGroupLReLU

    head_cls = BoxHeadHNM
    head_regressor_cls = L1Regressor

    @classmethod
    def get_ensembler_cls(cls, dim: int) -> Type[BaseEnsembler]:
        """
        Returns:
            Type[BaseEnsembler]: return class of ensembler to use for this
                class
        """
        return BoxEnsemblerSelectiveV2


@MODULE_REGISTRY.register
class RetinaUNetC016Focal(RetinaUNetCV001Focal):
    backbone_conv_cls = ConvInstanceLReLU
    neck_conv_cls = ConvInstanceLReLU
    head_conv_cls = ConvGroupLReLU

    head_cls = BoxHeadAll
    head_sampler_cls = None
    head_regressor_cls = L1Regressor
    head_classifier_cls = FocalClassifier

    @classmethod
    def get_ensembler_cls(cls, dim: int) -> Type[BaseEnsembler]:
        """
        Returns:
            Type[BaseEnsembler]: return class of ensembler to use for this
                class
        """
        return BoxEnsemblerSelectiveV2
