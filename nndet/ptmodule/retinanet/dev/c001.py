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

from nndet.arch.conv import ConvGroupRelu, ConvInstanceRelu
from nndet.arch.heads.classifier import BCECLassifier, FocalClassifier
from nndet.arch.heads.comb import BoxHeadAll, BoxHeadHNM
from nndet.arch.heads.regressor import L1Regressor
from nndet.core.boxes.matcher import ATSSMatcher
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.retinanet.base import RetinaNetModule


@MODULE_REGISTRY.register
class RetinaNetC001(RetinaNetModule):
    backbone_conv_cls = ConvInstanceRelu
    neck_conv_cls = ConvInstanceRelu
    head_conv_cls = ConvGroupRelu

    head_cls = BoxHeadHNM
    head_classifier_cls = BCECLassifier
    head_regressor_cls = L1Regressor
    matcher_cls = ATSSMatcher


@MODULE_REGISTRY.register
class RetinaNetC001Focal(RetinaNetModule):
    """
    Focal Loss based V001 RetinaNet
    (only intended for easy subclassing and not used in nnDetection V0.1)
    """

    head_cls = BoxHeadAll
    head_classifier_cls = FocalClassifier
    head_sampler_cls = None
