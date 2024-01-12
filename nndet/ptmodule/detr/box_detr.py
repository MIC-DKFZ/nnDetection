# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Type

from nndet.core.abstract import AbstractOneStageDetector
from nndet.core.boxes.criterions.base import BoxCriterion, ClassCriterion
from nndet.core.boxes.matcher1to1.base import BaseMatcher
from nndet.core.detr import BaseDETR
from nndet.core.post.detr import DETRBoxPost
from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.heads.classifier.ffn import FFNClassifier
from nndet.nn.heads.detr.base import DETRHead
from nndet.nn.heads.regressor.ffn import FFNRegressor
from nndet.nn.heads.segmenter import Segmenter
from nndet.nn.layers.pos_embed.base import BasePositionEmbedding
from nndet.nn.neck.abstract import AbstractNeck
from nndet.nn.neck.channel_mapper import ChannelMapper
from nndet.nn.transformer.abstract_transformer import AbstractTransformer
from nndet.nn.transformer.layers.abstract import (
    BaseTransformerDecoder,
    BaseTransformerEncoder,
)
from nndet.ptmodule.mixins.evaluation import BoxEvalMixin
from nndet.ptmodule.mixins.model.set import DETRModelMixin
from nndet.ptmodule.mixins.prediction import BoxPredictionMixin
from nndet.ptmodule.mixins.prepare import BoxesPrepareMixin
from nndet.ptmodule.module import LightningBaseModule
from nndet.utils.typing import CONVSEQ, LINEARSEQ


class BoxDETRModule(
    LightningBaseModule,  # Main module
    BoxesPrepareMixin,  # prepare batch for box training
    BoxEvalMixin,  # Bounding Box Evaluation
    DETRModelMixin,  # DETR Mixin to build the model
    BoxPredictionMixin,  # Bounding Box Sweep
):
    # define detector cls
    detector_cls: Type[AbstractOneStageDetector] = BaseDETR  #: define base detector class

    backbone_cls: Type[AbstractBackbone] = ...  #: define class for backbone
    backbone_conv_cls: Type[CONVSEQ] = ...  #: conv class used for backbone
    # Channel Mapper
    channel_mapper_cls: Type[ChannelMapper] = ...
    channel_mapper_conv_cls: Type[CONVSEQ] = ...

    # transformer
    pos_embed_cls: BasePositionEmbedding = ...
    transformer_encoder_cls: Type[BaseTransformerEncoder] = ...
    transformer_decoder_cls: Type[BaseTransformerDecoder] = ...
    transformer_cls: Type[AbstractTransformer] = ...

    # head blocks
    head_cls: DETRHead = ...  #: main DETR head
    head_linear_cls: LINEARSEQ = ...  #: conv class used for head
    head_classifier_cls: FFNClassifier = ...  #: define classifier class
    head_regressor_cls: FFNRegressor = ...  #: define regressor class
    head_box_post_cls: DETRBoxPost = ...  #: define postprocessing strategy during inference

    matcher_cls: BaseMatcher = ...  #: matching algorithm
    matcher_class_criterion_cls: ClassCriterion = ...  #: criterion to compute class cost matrix
    # either reg or box criterion need to be set
    # reg criterion usually operates on encoded targets while box cirterion operates on raw boxes
    # there is no structural difference though and just a nomenclature
    matcher_reg_criterion_cls: Optional[BoxCriterion] = None  #: criterion to compute regression cost matrix
    matcher_box_criterion_cls: Optional[BoxCriterion] = None  #: criterion to compute regression cost matrix

    # [Optional]
    neck_cls: Optional[Type[AbstractNeck]] = None  #: [optional] define class for neck
    neck_conv_cls: Optional[Type[CONVSEQ]] = None  #: [optional] conv class used for neck

    # [Optional] Semantic Segmenation Head
    segmenter_cls: Optional[Type[Segmenter]] = None  #: [optional] segmentation head
