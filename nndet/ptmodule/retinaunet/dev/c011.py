from loguru import logger
from nndet.arch.blocks.basic import MySEBlockExp2, MySEBlockExp4, StackedConvBlock2, StackedConvBlock3
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
    ConvGroupLReLU,
    Generator
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


@MODULE_REGISTRY.register
class RetinaUNetC011C3(RetinaUNetV001):
    block = StackedConvBlock3


@MODULE_REGISTRY.register
class RetinaUNetC011MySE2(RetinaUNetV001):
    block = MySEBlockExp2
    
    @classmethod
    def _build_encoder(
        cls,
        plan_arch: dict,
        model_cfg: dict,
    ):
        """
        Build encoder network

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            EncoderType: encoder instance
        """
        conv = Generator(cls.base_conv_cls, plan_arch["dim"])
        logger.info(f"Building:: encoder {cls.encoder_cls.__name__}: {model_cfg['encoder_kwargs']} ")
        encoder = cls.encoder_cls(
            conv=conv,
            conv_kernels=plan_arch["conv_kernels"],
            strides=plan_arch["strides"],
            block_cls=cls.block,
            in_channels=plan_arch["in_channels"],
            start_channels=plan_arch["start_channels"],
            stage_kwargs=None,
            max_channels=plan_arch.get("max_channels", 320),
            first_block_cls=StackedConvBlock2,
            **model_cfg['encoder_kwargs'],
        )
        return encoder

@MODULE_REGISTRY.register
class RetinaUNetC011MySE4(RetinaUNetV001):
    block = MySEBlockExp4
