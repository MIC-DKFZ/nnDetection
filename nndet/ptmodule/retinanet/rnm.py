# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Type

from nndet.core.abstract import AbstractOneStageDetector
from nndet.core.boxes.matcher import Matcher
from nndet.core.boxes.sampler import AbstractSampler
from nndet.core.post.box import BoxPostprocessing
from nndet.core.retina import BaseRetinaNet
from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.heads.classifier.dense import DenseClassifier
from nndet.nn.heads.comb.base import AnchorHead
from nndet.nn.heads.regressor.dense import DenseRegressor
from nndet.nn.neck.abstract import AbstractNeck
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.mixins.evaluation import BoxEvalMixin
from nndet.ptmodule.mixins.model import SingleStageMixin
from nndet.ptmodule.mixins.prediction.boxes import BoxPredictionMixinV2
from nndet.ptmodule.mixins.prepare import BoxesPrepareMixin
from nndet.ptmodule.module import LightningBaseModule
from nndet.utils.typing import CONVSEQ


@MODULE_REGISTRY.register
class RetinaNetModule(
    LightningBaseModule,  # Detection Base
    BoxesPrepareMixin,  # prepare batch for box training
    BoxEvalMixin,  # Boundig Box Evaluation
    SingleStageMixin,  # Single Stage Detector
    BoxPredictionMixinV2,  # Bounding Box Sweep
):
    # define detector cls
    detector_cls: Type[AbstractOneStageDetector] = BaseRetinaNet

    backbone_cls: Type[AbstractBackbone] = ...  # define class for backbone
    backbone_conv_cls: Type[CONVSEQ] = ...  # conv class used for backbone

    neck_cls: Type[AbstractNeck] = ...  # define class for neck
    neck_conv_cls: Type[CONVSEQ] = ...  # conv class used for neck

    head_cls: Type[AnchorHead] = ...  # define class for head
    head_conv_cls: Type[CONVSEQ] = ...  # conv class used for head
    head_classifier_cls: Type[DenseClassifier] = ...  # define class for head classifier
    head_regressor_cls: Type[DenseRegressor] = ...  # define class for head regressor
    # [optional] sampler class for negative mining
    # if None: no sampler will be given to the head
    head_sampler_cls: Optional[Type[AbstractSampler]] = ...

    matcher_cls: Type[Matcher] = ...  # define class to match anchors to ground truth
    box_post_cls: Type[BoxPostprocessing] = ...  # define box postprocessing strategy

    # Not suported here; See `RetinaUNet`
    segmenter_cls = None
