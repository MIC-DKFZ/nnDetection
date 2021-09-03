from typing import Callable, Hashable

from loguru import logger

from nndet.arch.conv import (
    ConvGroupLReLU,
    ConvGroupMish,
    ConvInstanceLReLU,
    ConvInstanceMish,
)
from nndet.arch.heads.classifier import FocalClassifier
from nndet.arch.heads.comb import BoxHeadAll, BoxHeadHNM
from nndet.arch.heads.regressor import L1Regressor
from nndet.inference.ensembler.detection import (
    BoxEnsemblerSelective2D,
    BoxEnsemblerSelectiveFaster,
)
from nndet.inference.ensembler.segmentation import SegmentationEnsembler
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.retinaunet.v001 import RetinaUNetCV001Focal, RetinaUNetV001


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
            },
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
            },
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


@MODULE_REGISTRY.register
class RetinaUNetC012FocalGpMish(RetinaUNetC012Focal):
    base_conv_cls = ConvGroupMish
    head_conv_cls = ConvGroupMish


@MODULE_REGISTRY.register
class RetinaUNetC012Ranger21(RetinaUNetC012):
    def configure_optimizers(self):
        from ranger21 import Ranger21

        # configure optimizer
        logger.info(
            f"Running: initial_lr {self.trainer_cfg['initial_lr']} "
            f"weight_decay {self.trainer_cfg['weight_decay']} "
            f"Ranger21"
        )
        optimizer = Ranger21(
            self.parameters(),
            lr=self.trainer_cfg["initial_lr"],
            weight_decay=self.trainer_cfg["weight_decay"],
            use_cheb=False,
            lookahead_active=True,
            normloss_active=True,
            normloss_factor=6e-4,
            use_adaptive_gradient_clipping=True,
            agc_clipping_value=0.01,
            use_madgrad=False,
            warmdown_active=True,
            num_warmup_iterations=None,
            num_epochs=self.train_epochs,
            num_batches_per_epoch=self.trainer_cfg["num_train_batches_per_epoch"],
            warmup_pct_default=0.3,
            using_gc=True,
        )
        optimizer.show_settings()
        return optimizer
