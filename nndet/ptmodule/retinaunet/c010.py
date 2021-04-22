from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001
from nndet.ptmodule import MODULE_REGISTRY

from nndet.models.heads.comb import BoxHeadNoSampler
from nndet.models.heads.classifier import (
    FocalClassifier,
    AsymmetricFocalClassifier,
    FullyConntectedBCECLassifier,
    )
from nndet.models.heads.segmenter import DiceTopKSegmenterFgBg
from nndet.models.conv import (
    ConvGroupRelu,
    ConvInstanceMish,
    ConvInstanceSwish,
    ConvGroupMish,
    ConvGroupSwish,
    ConvInstanceSiLU,
    ConvGroupSiLU,
    ConvInstanceLReLU,
    ConvGroupLReLU
    )


import torch
from loguru import logger
from nndet.training import optimizer

from nndet.training.optimizer import get_params_no_wd_on_norm
from nndet.training.learning_rate import LinearWarmupPolyLR


@MODULE_REGISTRY.register
class RetinaUNetC010(RetinaUNetV001):
    pass


@MODULE_REGISTRY.register
class RetinaUNetC010Focal(RetinaUNetC010):
    head_cls = BoxHeadNoSampler
    head_classifier_cls = FocalClassifier


@MODULE_REGISTRY.register
class RetinaUNetC010AsymFocal(RetinaUNetC010):
    head_cls = BoxHeadNoSampler
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
class RetinaUNetC010LReLUFocal(RetinaUNetC010):
    base_conv_cls = ConvInstanceLReLU
    head_conv_cls = ConvGroupLReLU

    head_cls = BoxHeadNoSampler
    head_classifier_cls = FocalClassifier


@MODULE_REGISTRY.register
class RetinaUNetC010LReLUAsymFocal(RetinaUNetC010):
    base_conv_cls = ConvInstanceLReLU
    head_conv_cls = ConvGroupLReLU

    head_cls = BoxHeadNoSampler
    head_classifier_cls = AsymmetricFocalClassifier


@MODULE_REGISTRY.register
class RetinaUNetC010FocalResLike(RetinaUNetC010Focal):
    @classmethod
    def _build_encoder(cls, conv, plan_arch, model_cfg) -> torch.nn.Module:
        from nndet.models.encoder.res import ResModularExp
        from nndet.models.blocks.res import ResBottleneck

        logger.info(f"Building:: encoder {cls.encoder_cls.__name__}: {model_cfg['encoder_kwargs']} ")
        return ResModularExp(
            conv=conv,
            conv_kernels=plan_arch["conv_kernels"],
            strides=plan_arch["strides"],

            num_blocks=[4, 12, 16, 8],
            block_cls=ResBottleneck,

            in_channels=plan_arch["in_channels"],
            start_channels=plan_arch["start_channels"],

            stage_kwargs=None,
            **model_cfg['encoder_kwargs'],
            expansion=4,
        )


@MODULE_REGISTRY.register
class RetinaUNetC010LK(RetinaUNetC010):
    def configure_optimizers(self):
        try:
            import torch_optimizer as optim
        except ImportError:
            raise ImportError("torch_optimizer needs to be installed to run this module."
                              "Please refer to https://github.com/jettify/pytorch-optimizer"
                              "to install it")

        # configure optimizer
        logger.info(f"Running: initial_lr {self.trainer_cfg['initial_lr']} "
                    f"weight_decay {self.trainer_cfg['weight_decay']} "
                    f"SGD Lookahead with momentum {self.trainer_cfg['sgd_momentum']} and "
                    f"nesterov {self.trainer_cfg['sgd_nesterov']}")
        wd_groups = get_params_no_wd_on_norm(self, weight_decay=self.trainer_cfg['weight_decay'])
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
        num_iterations = self.trainer_cfg["max_num_epochs"] * \
            self.trainer_cfg["num_train_batches_per_epoch"]
        scheduler = LinearWarmupPolyLR(
            optimizer=optimizer,
            warm_iterations=self.trainer_cfg["warm_iterations"],
            warm_lr=self.trainer_cfg["warm_lr"],
            poly_gamma=self.trainer_cfg["poly_gamma"],
            num_iterations=num_iterations
        )
        return [optimizer] , {'scheduler': scheduler, 'interval': 'step'}


@MODULE_REGISTRY.register
class RetinaUNetC010AdamW(RetinaUNetC010):
    def configure_optimizers(self):
        # configure optimizer
        logger.info(f"Running: initial_lr {self.trainer_cfg['initial_lr']} "
                    f"weight_decay {self.trainer_cfg['weight_decay']} "
                    f"AdamW")
        wd_groups = get_params_no_wd_on_norm(self, weight_decay=self.trainer_cfg['weight_decay'])
        optimizer = torch.optim.AdamW(
            wd_groups,
            self.trainer_cfg["initial_lr"],
            weight_decay=self.trainer_cfg["weight_decay"],
            )

        # configure lr scheduler
        num_iterations = self.trainer_cfg["max_num_epochs"] * \
            self.trainer_cfg["num_train_batches_per_epoch"]
        scheduler = LinearWarmupPolyLR(
            optimizer=optimizer,
            warm_iterations=self.trainer_cfg["warm_iterations"],
            warm_lr=self.trainer_cfg["warm_lr"],
            poly_gamma=self.trainer_cfg["poly_gamma"],
            num_iterations=num_iterations
        )
        return [optimizer] , {'scheduler': scheduler, 'interval': 'step'}


@MODULE_REGISTRY.register
class RetinaUNetC010RAdam(RetinaUNetC010):
    def configure_optimizers(self):
        try:
            import torch_optimizer as optim
        except ImportError:
            raise ImportError("torch_optimizer needs to be installed to run this module."
                              "Please refer to https://github.com/jettify/pytorch-optimizer"
                              "to install it")

        # configure optimizer
        logger.info(f"Running: initial_lr {self.trainer_cfg['initial_lr']} "
                    f"weight_decay {self.trainer_cfg['weight_decay']} "
                    f"RAdam")
        wd_groups = get_params_no_wd_on_norm(self, weight_decay=self.trainer_cfg['weight_decay'])
        optimizer = optim.RAdam(
            wd_groups,
            lr=self.trainer_cfg["initial_lr"],
            weight_decay=self.trainer_cfg["weight_decay"],
            )

        # configure lr scheduler
        num_iterations = self.trainer_cfg["max_num_epochs"] * \
            self.trainer_cfg["num_train_batches_per_epoch"]
        scheduler = LinearWarmupPolyLR(
            optimizer=optimizer,
            warm_iterations=self.trainer_cfg["warm_iterations"],
            warm_lr=self.trainer_cfg["warm_lr"],
            poly_gamma=self.trainer_cfg["poly_gamma"],
            num_iterations=num_iterations
        )
        return [optimizer] , {'scheduler': scheduler, 'interval': 'step'}


@MODULE_REGISTRY.register
class RetinaUNetC010Ranger(RetinaUNetC010):
    def configure_optimizers(self):
        try:
            import torch_optimizer as optim
        except ImportError:
            raise ImportError("torch_optimizer needs to be installed to run this module."
                              "Please refer to https://github.com/jettify/pytorch-optimizer"
                              "to install it")

        # configure optimizer
        logger.info(f"Running: initial_lr {self.trainer_cfg['initial_lr']} "
                    f"weight_decay {self.trainer_cfg['weight_decay']} "
                    f"Ranger")
        wd_groups = get_params_no_wd_on_norm(self, weight_decay=self.trainer_cfg['weight_decay'])
        optimizer = optim.Ranger(
            wd_groups,
            self.trainer_cfg["initial_lr"],
            weight_decay=self.trainer_cfg["weight_decay"],
            )

        # configure lr scheduler
        num_iterations = self.trainer_cfg["max_num_epochs"] * \
            self.trainer_cfg["num_train_batches_per_epoch"]
        scheduler = LinearWarmupPolyLR(
            optimizer=optimizer,
            warm_iterations=self.trainer_cfg["warm_iterations"],
            warm_lr=self.trainer_cfg["warm_lr"],
            poly_gamma=self.trainer_cfg["poly_gamma"],
            num_iterations=num_iterations
        )
        return [optimizer] , {'scheduler': scheduler, 'interval': 'step'}


@MODULE_REGISTRY.register
class RetinaUNetC010Madgrad(RetinaUNetC010):
    def configure_optimizers(self):
        try:
            from madgrad import MADGRAD
        except ImportError:
            raise ImportError("madgrad needs to be installed to run this module."
                              "Please refer to https://github.com/facebookresearch/madgrad"
                              "to install it")

        # configure optimizer
        logger.info(f"Running: initial_lr {self.trainer_cfg['initial_lr']} "
                    f"weight_decay {self.trainer_cfg['weight_decay']} "
                    f"MADGRAD with momentum {self.trainer_cfg['momentum']}")
        wd_groups = get_params_no_wd_on_norm(self, weight_decay=self.trainer_cfg['weight_decay'])
        optimizer = MADGRAD(
            wd_groups,
            self.trainer_cfg["initial_lr"],
            weight_decay=self.trainer_cfg["weight_decay"],
            momentum=self.trainer_cfg["momentum"],
            )

        # configure lr scheduler
        num_iterations = self.trainer_cfg["max_num_epochs"] * \
            self.trainer_cfg["num_train_batches_per_epoch"]
        scheduler = LinearWarmupPolyLR(
            optimizer=optimizer,
            warm_iterations=self.trainer_cfg["warm_iterations"],
            warm_lr=self.trainer_cfg["warm_lr"],
            poly_gamma=self.trainer_cfg["poly_gamma"],
            num_iterations=num_iterations
        )
        return [optimizer], {'scheduler': scheduler, 'interval': 'step'}


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
class RetinaUNetC010GNMishAllFocal(RetinaUNetC010):
    base_conv_cls = ConvGroupMish
    head_conv_cls = ConvGroupMish

    head_cls = BoxHeadNoSampler
    head_classifier_cls = FocalClassifier


@MODULE_REGISTRY.register
class RetinaUNetC010TopK10FGBG(RetinaUNetC010):
    segmenter_cls = DiceTopKSegmenterFgBg


@MODULE_REGISTRY.register
class RetinaUNetC010TopK10FGBGMad(RetinaUNetC010Madgrad):
    segmenter_cls = DiceTopKSegmenterFgBg
