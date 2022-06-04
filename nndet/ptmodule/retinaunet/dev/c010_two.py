# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from nndet.core.boxes.matcher import ATSSMatcher, IoUMatcher
from nndet.nn.heads.classifier.dense import BCECLassifier, FocalClassifier
from nndet.nn.heads.comb.anchor_all import BoxHeadAll
from nndet.nn.heads.comb.anchor_sampled import BoxHeadHNMNative
from nndet.nn.heads.regressor.dense import GIoURegressor
from nndet.nn.heads.segmenter import DiCESegmenterFgBg
from nndet.nn.layers.conv import ConvGroupRelu, ConvInstanceRelu
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.retinaunet.dev.c010 import RetinaUNetC010LReLU


@MODULE_REGISTRY.register
class RetinaUNetC010Two(RetinaUNetC010LReLU):
    base_conv_cls = ConvInstanceRelu
    head_conv_cls = ConvGroupRelu

    head_cls = BoxHeadHNMNative
    head_classifier_cls = BCECLassifier
    head_regressor_cls = GIoURegressor
    matcher_cls = IoUMatcher
    segmenter_cls = DiCESegmenterFgBg


@MODULE_REGISTRY.register
class RetinaUNetC010TwoATSS(RetinaUNetC010LReLU):
    base_conv_cls = ConvInstanceRelu
    head_conv_cls = ConvGroupRelu

    head_cls = BoxHeadHNMNative
    head_classifier_cls = BCECLassifier
    head_regressor_cls = GIoURegressor
    matcher_cls = ATSSMatcher
    segmenter_cls = DiCESegmenterFgBg


@MODULE_REGISTRY.register
class RetinaUNetC010TwoFocal(RetinaUNetC010LReLU):
    base_conv_cls = ConvInstanceRelu
    head_conv_cls = ConvGroupRelu

    head_cls = BoxHeadAll
    head_classifier_cls = FocalClassifier
    head_regressor_cls = GIoURegressor
    matcher_cls = IoUMatcher
    segmenter_cls = DiCESegmenterFgBg

    @classmethod
    def _build_head(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        classifier,
        regressor,
        coder,
    ):
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
        head_kwargs = model_cfg["head_kwargs"]

        head = cls.head_cls(
            classifier=classifier,
            regressor=regressor,
            coder=coder,
            **head_kwargs,
        )
        return head


@MODULE_REGISTRY.register
class RetinaUNetC010TwoFocalATSS(RetinaUNetC010TwoFocal):
    base_conv_cls = ConvInstanceRelu
    head_conv_cls = ConvGroupRelu

    head_cls = BoxHeadAll
    head_classifier_cls = FocalClassifier
    head_regressor_cls = GIoURegressor
    matcher_cls = ATSSMatcher
    segmenter_cls = DiCESegmenterFgBg
