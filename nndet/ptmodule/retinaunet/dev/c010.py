import torch
from loguru import logger

from nndet.core.boxes.coder import CoderType
from nndet.nn.conv import (
    ConvGroupLReLU,
    ConvGroupMish,
    ConvGroupRelu,
    ConvGroupSiLU,
    ConvGroupSwish,
    ConvInstanceLReLU,
    ConvInstanceMish,
    ConvInstanceSiLU,
    ConvInstanceSwish,
)
from nndet.nn.decoder.base import SmallerUFPN, SmallUFPN
from nndet.nn.heads.classifier import (
    AsymmetricFocalClassifier,
    DenseClassifierType,
    FocalClassifier,
    FullyConntectedBCECLassifier,
)
from nndet.nn.heads.comb import AnchorHeadType, BoxHeadAll
from nndet.nn.heads.regressor import DenseRegressorType
from nndet.nn.heads.segmenter import DiceTopKSegmenterFgBg
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.retinaunet.runv001 import RetinaUNetV001
from nndet.training.learning_rate import LinearWarmupPolyLR
from nndet.training.optimizer import get_params_no_wd_on_norm


@MODULE_REGISTRY.register
class RetinaUNetC010(RetinaUNetV001):
    pass


@MODULE_REGISTRY.register
class RetinaUNetC010Focal(RetinaUNetC010):
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
        sampler_name = cls.head_sampler_cls.__name__
        sampler_kwargs = model_cfg["head_sampler_kwargs"]

        logger.info(
            f"Building:: head {head_name}: {head_kwargs} "
            f"sampler {sampler_name}: {sampler_kwargs}"
        )
        head = cls.head_cls(
            classifier=classifier,
            regressor=regressor,
            coder=coder,
            **head_kwargs,
        )
        return head


@MODULE_REGISTRY.register
class RetinaUNetC010AsymFocal(RetinaUNetC010Focal):
    head_cls = BoxHeadAll
    head_classifier_cls = AsymmetricFocalClassifier


@MODULE_REGISTRY.register
class RetinaUNetC010Mish(RetinaUNetC010):
    base_conv_cls = ConvInstanceMish
    head_conv_cls = ConvGroupMish


@MODULE_REGISTRY.register
class RetinaUNetC010Swish(RetinaUNetC010):
    base_conv_cls = ConvInstanceSwish
    head_conv_cls = ConvGroupSwish


@MODULE_REGISTRY.register
class RetinaUNetC010SiLU(RetinaUNetC010):
    base_conv_cls = ConvInstanceSiLU
    head_conv_cls = ConvGroupSiLU


@MODULE_REGISTRY.register
class RetinaUNetC010LReLU(RetinaUNetC010):
    base_conv_cls = ConvInstanceLReLU
    head_conv_cls = ConvGroupLReLU


@MODULE_REGISTRY.register
class RetinaUNetC010LReLUFocal(RetinaUNetC010Focal):
    base_conv_cls = ConvInstanceLReLU
    head_conv_cls = ConvGroupLReLU

    head_cls = BoxHeadAll
    head_classifier_cls = FocalClassifier


@MODULE_REGISTRY.register
class RetinaUNetC010LReLUAsymFocal(RetinaUNetC010Focal):
    base_conv_cls = ConvInstanceLReLU
    head_conv_cls = ConvGroupLReLU

    head_cls = BoxHeadAll
    head_classifier_cls = AsymmetricFocalClassifier


@MODULE_REGISTRY.register
class RetinaUNetC010LK(RetinaUNetC010):
    def configure_optimizers(self):
        try:
            import torch_optimizer as optim
        except ImportError:
            raise ImportError(
                "torch_optimizer needs to be installed to run this module."
                "Please refer to https://github.com/jettify/pytorch-optimizer"
                "to install it"
            )

        # configure optimizer
        logger.info(
            f"Running: initial_lr {self.trainer_cfg['initial_lr']} "
            f"weight_decay {self.trainer_cfg['weight_decay']} "
            f"SGD Lookahead with momentum {self.trainer_cfg['sgd_momentum']} and "
            f"nesterov {self.trainer_cfg['sgd_nesterov']}"
        )
        wd_groups = get_params_no_wd_on_norm(
            self, weight_decay=self.trainer_cfg["weight_decay"]
        )
        _optimizer = torch.optim.SGD(
            wd_groups,
            self.trainer_cfg["initial_lr"],
            weight_decay=self.trainer_cfg["weight_decay"],
            momentum=self.trainer_cfg["sgd_momentum"],
            nesterov=self.trainer_cfg["sgd_nesterov"],
        )
        optimizer = optim.Lookahead(
            _optimizer,
            k=5,
            alpha=0.5,
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


@MODULE_REGISTRY.register
class RetinaUNetC010AdamW(RetinaUNetC010):
    def configure_optimizers(self):
        # configure optimizer
        logger.info(
            f"Running: initial_lr {self.trainer_cfg['initial_lr']} "
            f"weight_decay {self.trainer_cfg['weight_decay']} "
            f"AdamW"
        )
        wd_groups = get_params_no_wd_on_norm(
            self, weight_decay=self.trainer_cfg["weight_decay"]
        )
        optimizer = torch.optim.AdamW(
            wd_groups,
            self.trainer_cfg["initial_lr"],
            weight_decay=self.trainer_cfg["weight_decay"],
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


@MODULE_REGISTRY.register
class RetinaUNetC010RAdam(RetinaUNetC010):
    def configure_optimizers(self):
        try:
            import torch_optimizer as optim
        except ImportError:
            raise ImportError(
                "torch_optimizer needs to be installed to run this module."
                "Please refer to https://github.com/jettify/pytorch-optimizer"
                "to install it"
            )

        # configure optimizer
        logger.info(
            f"Running: initial_lr {self.trainer_cfg['initial_lr']} "
            f"weight_decay {self.trainer_cfg['weight_decay']} "
            f"RAdam"
        )
        wd_groups = get_params_no_wd_on_norm(
            self, weight_decay=self.trainer_cfg["weight_decay"]
        )
        optimizer = optim.RAdam(
            wd_groups,
            lr=self.trainer_cfg["initial_lr"],
            weight_decay=self.trainer_cfg["weight_decay"],
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


@MODULE_REGISTRY.register
class RetinaUNetC010Ranger(RetinaUNetC010):
    def configure_optimizers(self):
        try:
            import torch_optimizer as optim
        except ImportError:
            raise ImportError(
                "torch_optimizer needs to be installed to run this module."
                "Please refer to https://github.com/jettify/pytorch-optimizer"
                "to install it"
            )

        # configure optimizer
        logger.info(
            f"Running: initial_lr {self.trainer_cfg['initial_lr']} "
            f"weight_decay {self.trainer_cfg['weight_decay']} "
            f"Ranger"
        )
        wd_groups = get_params_no_wd_on_norm(
            self, weight_decay=self.trainer_cfg["weight_decay"]
        )
        optimizer = optim.Ranger(
            wd_groups,
            self.trainer_cfg["initial_lr"],
            weight_decay=self.trainer_cfg["weight_decay"],
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


@MODULE_REGISTRY.register
class RetinaUNetC010Madgrad(RetinaUNetC010):
    def configure_optimizers(self):
        try:
            from madgrad import MADGRAD
        except ImportError:
            raise ImportError(
                "madgrad needs to be installed to run this module."
                "Please refer to https://github.com/facebookresearch/madgrad"
                "to install it"
            )

        # configure optimizer
        logger.info(
            f"Running: initial_lr {self.trainer_cfg['initial_lr']} "
            f"weight_decay {self.trainer_cfg['weight_decay']} "
            f"MADGRAD with momentum {self.trainer_cfg['momentum']}"
        )
        wd_groups = get_params_no_wd_on_norm(
            self, weight_decay=self.trainer_cfg["weight_decay"]
        )
        optimizer = MADGRAD(
            wd_groups,
            self.trainer_cfg["initial_lr"],
            weight_decay=self.trainer_cfg["weight_decay"],
            momentum=self.trainer_cfg["momentum"],
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


@MODULE_REGISTRY.register
class RetinaUNetC010DoubleHead(RetinaUNetC010):
    head_classifier_cls = FullyConntectedBCECLassifier


@MODULE_REGISTRY.register
class RetinaUNetC010GNAll(RetinaUNetC010):
    base_conv_cls = ConvGroupRelu
    head_conv_cls = ConvGroupRelu


@MODULE_REGISTRY.register
class RetinaUNetC010GNMishAll(RetinaUNetC010):
    base_conv_cls = ConvGroupMish
    head_conv_cls = ConvGroupMish


@MODULE_REGISTRY.register
class RetinaUNetC010GNMishAllFocal(RetinaUNetC010Focal):
    base_conv_cls = ConvGroupMish
    head_conv_cls = ConvGroupMish

    head_cls = BoxHeadAll
    head_classifier_cls = FocalClassifier


@MODULE_REGISTRY.register
class RetinaUNetC010TopK10FGBG(RetinaUNetC010):
    segmenter_cls = DiceTopKSegmenterFgBg


@MODULE_REGISTRY.register
class RetinaUNetC010TopK10FGBGMad(RetinaUNetC010Madgrad):
    segmenter_cls = DiceTopKSegmenterFgBg


@MODULE_REGISTRY.register
class RetinaUNetC010LReLUMishHead(RetinaUNetC010LReLU):
    head_conv_cls = ConvGroupMish


@MODULE_REGISTRY.register
class RetinaUNetC010LReLUSmallU(RetinaUNetC010LReLU):
    decoder_cls = SmallUFPN


@MODULE_REGISTRY.register
class RetinaUNetC010LReLUSmallerU(RetinaUNetC010LReLU):
    decoder_cls = SmallerUFPN
