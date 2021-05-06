from loguru import logger
from nndet.arch.heads.comb.anchor_sampled import BoxHeadHNMDualReg, BoxHeadHNMRegAll

from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001
from nndet.ptmodule import MODULE_REGISTRY
from nndet.arch.heads.comb.base import AnchorHeadType
from nndet.arch.heads.classifier.dense import DenseClassifierType
from nndet.arch.heads.regressor.dense_single import DenseRegressorType, DualRegressor
from nndet.core.boxes.coder import CoderType

from nndet.arch.heads.comb import (
    BoxHeadAll,
    BoxHeadHNM,
)
from nndet.arch.heads.classifier import (
    FocalClassifier,
    AsymmetricFocalClassifier,
    )
from nndet.arch.heads.regressor import (
    L1Regressor
    )
from nndet.arch.conv import (
    ConvInstanceLReLU,
    ConvGroupLReLU
    )


@MODULE_REGISTRY.register
class RetinaUNetC011(RetinaUNetV001):
    base_conv_cls = ConvInstanceLReLU
    head_conv_cls = ConvGroupLReLU


@MODULE_REGISTRY.register
class RetinaUNetC011L1(RetinaUNetC011):
    head_cls = BoxHeadHNM
    head_regressor_cls = L1Regressor


@MODULE_REGISTRY.register
class RetinaUNetC011L1All(RetinaUNetC011):
    head_cls = BoxHeadHNMRegAll
    head_regressor_cls = L1Regressor


@MODULE_REGISTRY.register
class RetinaUNetC011DualReg(RetinaUNetC011):
    head_cls = BoxHeadHNMDualReg
    head_regressor_cls = DualRegressor


@MODULE_REGISTRY.register
class RetinaUNetC011Focal(RetinaUNetC011):
    head_cls = BoxHeadAll
    head_classifier_cls = FocalClassifier

    @classmethod
    def _build_head(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        classifier: DenseClassifierType,
        regressor: DenseRegressorType,
        coder: CoderType,
    ) -> AnchorHeadType:
        """
        Build detection head

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings
            classifier: classifier instance
            regressor: regressor instance
            coder: coder instance to encode boxes

        Returns:
            HeadType: instantiated head
        """
        head_name = cls.head_cls.__name__
        head_kwargs = model_cfg['head_kwargs']
        sampler_name = cls.head_sampler_cls.__name__
        sampler_kwargs = model_cfg['head_sampler_kwargs']

        logger.info(f"Building:: head {head_name}: {head_kwargs} "
                    f"sampler {sampler_name}: {sampler_kwargs}")
        head = cls.head_cls(
            classifier=classifier,
            regressor=regressor,
            coder=coder,
            **head_kwargs,
        )
        return head


@MODULE_REGISTRY.register
class RetinaUNetC011AsymFocal(RetinaUNetC011Focal):
    head_cls = BoxHeadAll
    head_classifier_cls = AsymmetricFocalClassifier
