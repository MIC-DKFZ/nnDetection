from typing import Callable, Hashable

from nndet.inference.ensembler.segmentation import SegmentationEnsembler
from nndet.inference.ensembler.detection import (
    BoxEnsemblerSelectiveFaster,
    BoxEnsemblerSelective2D,
    )

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
    ConvInstanceMish,
    ConvGroupMish,
)


@MODULE_REGISTRY.register
class RetinaUNetC012(RetinaUNetV001):
    base_conv_cls = ConvInstanceLReLU
    head_conv_cls = ConvGroupLReLU

    head_cls = BoxHeadHNM
    head_regressor_cls = L1Regressor

    @staticmethod
    def get_ensembler_cls(key: Hashable, dim: int) -> Callable:
        """
        Get ensembler classes to combine multiple predictions
        Needs to be overwritten in subclasses!
        """
        _lookup = {
            2: {
                "boxes": BoxEnsemblerSelective2D,
                "seg": SegmentationEnsembler,
            },
            3: {
                "boxes": BoxEnsemblerSelectiveFaster,
                "seg": SegmentationEnsembler,
            }
        }
        if dim == 2:
            raise NotImplementedError
        return _lookup[dim][key]


@MODULE_REGISTRY.register
class RetinaUNetC012Focal(RetinaUNetCV001Focal):
    base_conv_cls = ConvInstanceLReLU
    head_conv_cls = ConvGroupLReLU

    head_cls = BoxHeadAll
    head_regressor_cls = L1Regressor
    head_classifier_cls = FocalClassifier

    @staticmethod
    def get_ensembler_cls(key: Hashable, dim: int) -> Callable:
        """
        Get ensembler classes to combine multiple predictions
        Needs to be overwritten in subclasses!
        """
        _lookup = {
            2: {
                "boxes": BoxEnsemblerSelective2D,
                "seg": SegmentationEnsembler,
            },
            3: {
                "boxes": BoxEnsemblerSelectiveFaster,
                "seg": SegmentationEnsembler,
            }
        }
        if dim == 2:
            raise NotImplementedError
        return _lookup[dim][key]


@MODULE_REGISTRY.register
class RetinaUNetC012Mish(RetinaUNetC012):
    base_conv_cls = ConvInstanceMish
    head_conv_cls = ConvGroupMish


@MODULE_REGISTRY.register
class RetinaUNetC012FocalMish(RetinaUNetC012Focal):
    base_conv_cls = ConvInstanceMish
    head_conv_cls = ConvGroupMish
