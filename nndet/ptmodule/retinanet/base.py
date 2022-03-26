# Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Optional, Type

from nndet.arch.blocks.basic import AbstractBlock
from nndet.arch.conv import ConvSeq
from nndet.arch.decoder.base import BaseUFPN
from nndet.arch.encoder.abstract import AbstractEncoder
from nndet.arch.heads.classifier.dense import DenseClassifier
from nndet.arch.heads.comb.base import AnchorHead
from nndet.arch.heads.regressor.dense import DenseRegressor
from nndet.core.abstract import AbstractOneStageDetector
from nndet.core.boxes.matcher import Matcher
from nndet.core.boxes.sampler import SamplerType
from nndet.core.retina import BaseRetinaNet
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.mixins.evaluation import BoxEvalMixin
from nndet.ptmodule.mixins.model import SingleStageMixin
from nndet.ptmodule.mixins.prediction import BoxPredictionMixin
from nndet.ptmodule.mixins.prepare import BoxesPrepareMixin
from nndet.ptmodule.module import LightningBaseModule


@MODULE_REGISTRY.register
class RetinaNetModule(
    LightningBaseModule,  # Detection Base
    BoxesPrepareMixin,  # prepare batch for box training
    BoxEvalMixin,  # Boundig Box Evaluation
    SingleStageMixin,  # Single Stage Detector
    BoxPredictionMixin,  # Bounding Box Sweep
):
    # define detector cls
    detector_cls: Type[AbstractOneStageDetector] = BaseRetinaNet

    backbone_cls: Type[AbstractEncoder] = ...  # define class for backbone
    backbone_conv_cls: Type[ConvSeq] = ...  # conv class used for backbone
    backbone_block: Type[
        AbstractBlock
    ] = ...  # define central building block of backbone

    neck_cls: Type[BaseUFPN] = ...  # define class for neck
    neck_conv_cls: Type[ConvSeq] = ...  # conv class used for neck

    head_cls: Type[AnchorHead] = ...  # define class for head
    head_conv_cls: Type[ConvSeq] = ...  # conv class used for head
    head_classifier_cls: Type[DenseClassifier] = ...  # define class for head classifier
    head_regressor_cls: Type[DenseRegressor] = ...  # define class for head regressor
    # [optional] sampler class for negative mining
    # if None: no sampler will be given to the head
    head_sampler_cls: Optional[Type[SamplerType]] = ...

    matcher_cls: Type[Matcher] = ...  # define class to match anchors to ground truth
    # Not suported here; See `RetinaUNet`
    segmenter_cls = None
