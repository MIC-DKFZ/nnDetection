import copy
from loguru import logger
from nndet.arch.heads.comb.anchor_sampled import BoxHeadHNMDualReg, BoxHeadHNMRegAll

from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001
from nndet.ptmodule import MODULE_REGISTRY
from nndet.arch.heads.comb.base import AnchorHeadType
from nndet.arch.heads.classifier.dense import DenseClassifierType
from nndet.arch.heads.regressor.dense_single import DenseRegressorType, DualRegressor
from nndet.core.boxes.coder import CoderType
from nndet.arch.blocks.basic import StackedConvBlock2Max
from nndet.training.ema import EMAWeightsCB
from nndet.training.learning_rate import LinearWarmupPolyLR
from nndet.training.optimizer.sam import SAM

from nndet.arch.blocks.basic import (
    MySEBlockExp2,
    MySEBlockExp4,
    StackedConvBlock2,
    StackedConvBlock3,
    StackedResPlain,
)
from nndet.arch.heads.comb import (
    BoxHeadAll,
    BoxHeadHNM,
)
from nndet.arch.heads.classifier import (
    FocalClassifier,
    AsymmetricFocalClassifier,
)
from nndet.arch.heads.regressor import L1Regressor
from nndet.arch.conv import ConvInstanceLReLU, ConvGroupLReLU, Generator
from nndet.training.optimizer.utils import get_params_no_wd_on_norm
import torch


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
class RetinaUNetC011L1MaxF(RetinaUNetC011L1):
    block = StackedConvBlock2Max


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
        head_kwargs = model_cfg["head_kwargs"]

        logger.info(f"Building:: head {head_name}: {head_kwargs}")
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
class RetinaUNetC011C3AsymFocal(RetinaUNetC011AsymFocal):
    block = StackedConvBlock3


@MODULE_REGISTRY.register
class RetinaUNetC011C3Focal(RetinaUNetC011Focal):
    block = StackedConvBlock3


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
        _kwargs = copy.deepcopy(model_cfg["encoder_kwargs"])
        num_blocks = _kwargs.pop("num_blocks", None)
        if num_blocks is not None:
            i = len(plan_arch["conv_kernels"]) - 1
            _kwargs["stage_kwargs"] = [{"num_blocks": 1}] + [
                {"num_blocks": num_blocks}
            ] * i

        conv = Generator(cls.base_conv_cls, plan_arch["dim"])
        logger.info(
            f"Building:: encoder {cls.encoder_cls.__name__}: {model_cfg['encoder_kwargs']} "
        )
        encoder = cls.encoder_cls(
            conv=conv,
            conv_kernels=plan_arch["conv_kernels"],
            strides=plan_arch["strides"],
            block_cls=cls.block,
            in_channels=plan_arch["in_channels"],
            start_channels=plan_arch["start_channels"],
            max_channels=plan_arch.get("max_channels", 320),
            first_block_cls=StackedConvBlock2,
            **_kwargs,
        )
        return encoder


@MODULE_REGISTRY.register
class RetinaUNetC011MySE4(RetinaUNetV001):
    block = MySEBlockExp4


@MODULE_REGISTRY.register
class RetinaUNetC011ResPlain(RetinaUNetC011MySE2):
    block = StackedResPlain


@MODULE_REGISTRY.register
class RetinaUNetC011L1EMA(RetinaUNetC011):
    """
    Note: This subclasses the wrong class and is actually not computed with L1
    """

    def configure_callbacks(self):
        logger.warning("This implementation does not work with Multi-GPU!")
        callbacks = super().configure_callbacks()

        callbacks.append(
            EMAWeightsCB(
                device="cpu",
                beta=self.trainer_cfg["ema_beta"],
                ema_eval=False,
                dirpath="./",  # FIXME
            )
        )
        return callbacks


@MODULE_REGISTRY.register
class RetinaUNetC011L1SAM(RetinaUNetC011L1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False
        logger.warning("This implementation does not work with Multi-GPU!")

    def training_step(self, batch, batch_idx):
        """
        Computes a single training step
        See :class:`BaseRetinaNet` for more information
        """
        optimizer = self.optimizers()

        with torch.no_grad():
            batch = self.pre_trafo(**batch)

        # first step
        losses, _ = self.model.train_step(
            images=batch["data"],
            targets={
                "target_boxes": batch["boxes"],
                "target_classes": batch["classes"],
                "target_seg": batch["target"][:, 0],  # Remove channel dimension
            },
            predict=False,
            batch_num=batch_idx,
        )
        loss = sum(losses.values())
        self.manual_backward(loss)
        optimizer.first_step(zero_grad=True)

        # second step
        _losses, _ = self.model.train_step(
            images=batch["data"],
            targets={
                "target_boxes": batch["boxes"],
                "target_classes": batch["classes"],
                "target_seg": batch["target"][:, 0],  # Remove channel dimension
            },
            predict=False,
            batch_num=batch_idx,
        )
        _loss = sum(_losses.values())
        self.manual_backward(_loss)
        optimizer.second_step(zero_grad=True)
        return {"loss": loss, **{key: l.detach().item() for key, l in losses.items()}}

    def configure_optimizers(self):
        # configure optimizer
        logger.info(
            f"Running: initial_lr {self.trainer_cfg['initial_lr']} "
            f"weight_decay {self.trainer_cfg['weight_decay']} "
            f"SGD SAM with momentum {self.trainer_cfg['sgd_momentum']} and "
            f"nesterov {self.trainer_cfg['sgd_nesterov']}"
        )
        wd_groups = get_params_no_wd_on_norm(
            self, weight_decay=self.trainer_cfg["weight_decay"]
        )

        optimizer = SAM(
            wd_groups,
            torch.optim.SGD,
            lr=self.trainer_cfg["initial_lr"],
            weight_decay=self.trainer_cfg["weight_decay"],
            momentum=self.trainer_cfg["sgd_momentum"],
            nesterov=self.trainer_cfg["sgd_nesterov"],
            rho=self.trainer_cfg["sam_rho"],
            adaptive=self.trainer_cfg["sam_adaptive"],
        )

        # configure lr scheduler
        num_iterations = (
            self.train_epochs * self.trainer_cfg["num_train_batches_per_epoch"]
        )
        scheduler = LinearWarmupPolyLR(
            optimizer=optimizer,
            warm_iterations=self.trainer_cfg["warm_iterations"],
            warm_lr=self.trainer_cfg["warm_lr"],
            poly_gamma=self.trainer_cfg["poly_gamma"],
            num_iterations=num_iterations,
        )
        return [optimizer], {"scheduler": scheduler, "interval": "step"}
