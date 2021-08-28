"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from loguru import logger

from nndet.arch.conv import ConvGroupRelu, ConvInstanceRelu
from nndet.arch.heads.classifier import BCECLassifier, FocalClassifier
from nndet.arch.heads.classifier.dense import DenseClassifierType
from nndet.arch.heads.comb import BoxHeadAll, BoxHeadHNMNative
from nndet.arch.heads.comb.base import AnchorHeadType
from nndet.arch.heads.regressor import GIoURegressor
from nndet.arch.heads.regressor.dense_single import DenseRegressorType
from nndet.arch.heads.segmenter import DiCESegmenterFgBg
from nndet.core.boxes.coder import CoderType
from nndet.core.boxes.matcher import ATSSMatcher
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.retinaunet.base import RetinaUNetModule


@MODULE_REGISTRY.register
class RetinaUNetV001(RetinaUNetModule):
    base_conv_cls = ConvInstanceRelu
    head_conv_cls = ConvGroupRelu

    head_cls = BoxHeadHNMNative
    head_classifier_cls = BCECLassifier
    head_regressor_cls = GIoURegressor
    matcher_cls = ATSSMatcher
    segmenter_cls = DiCESegmenterFgBg


@MODULE_REGISTRY.register
class RetinaUNetCV001Focal(RetinaUNetV001):
    """
    Focal Loss based V001 RetinaUNet
    (only intended for easy subclassing and not used in nnDetection V0.1)
    """

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
