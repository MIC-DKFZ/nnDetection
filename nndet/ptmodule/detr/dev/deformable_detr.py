# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Type

from nndet.core.boxes.criterions.base import ClassCriterion
from nndet.core.boxes.criterions.cls import (
    FocalClassCriterionSigmoid,
    SimpleClassCriterionSoftmax,
)
from nndet.core.post.detr import DETRBoxPost, MaxFGBoxPost, TopKBoxPost
from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.backbone.blueprints.conv import ConvBackbone
from nndet.nn.heads.classifier.ffn import (
    CEFFNClassifier,
    FFNClassifier,
    FocalFFNClassifier,
)
from nndet.nn.heads.detr.base import DETRHead
from nndet.nn.heads.detr.deformable_detr import DeformableDETRHead
from nndet.nn.heads.regressor.ffn import FFNRegressor, L1UGIoUFFNRegressor
from nndet.nn.layers.conv import ConvGroupRelu
from nndet.nn.neck.channel_mapper import ChannelMapper
from nndet.nn.transformer.abstract_transformer import AbstractTransformer
from nndet.nn.transformer.deformable_transformer import DeformableDETRTransformer
from nndet.nn.transformer.layers.abstract import (
    BaseTransformerDecoder,
    BaseTransformerEncoder,
)
from nndet.nn.transformer.layers.deformable_detr import (
    DeformableDETRTransformerDecoder,
    DeformableDETRTransformerEncoder,
)
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.detr.dev.c001 import BoxDETRC001
from nndet.ptmodule.mixins.model.deformable_detr import DeformableSetModelMixin
from nndet.utils.typing import CONVSEQ


@MODULE_REGISTRY.register
class BoxDeformableDETRC001Focal(DeformableSetModelMixin, BoxDETRC001):
    backbone_cls: Type[AbstractBackbone] = ConvBackbone  #: define class for backbone

    channel_mapper_cls: Type[ChannelMapper] = ChannelMapper
    channel_mapper_conv_cls: Type[CONVSEQ] = ConvGroupRelu

    transformer_encoder_cls: BaseTransformerEncoder = DeformableDETRTransformerEncoder
    transformer_decoder_cls: BaseTransformerDecoder = DeformableDETRTransformerDecoder
    transformer_cls: AbstractTransformer = DeformableDETRTransformer

    head_cls: DETRHead = DeformableDETRHead  #: main DETR head
    head_classifier_cls: FFNClassifier = FocalFFNClassifier  #: define classifier class
    head_regressor_cls: FFNRegressor = L1UGIoUFFNRegressor  #: define regressor class
    head_box_post_cls: DETRBoxPost = TopKBoxPost  #: define postprocessing strategy during inference
    matcher_class_criterion_cls: ClassCriterion = FocalClassCriterionSigmoid  #: criterion to compute class cost matrix


@MODULE_REGISTRY.register
class BoxDeformableDETRC001CE(BoxDeformableDETRC001Focal):
    head_classifier_cls: FFNClassifier = CEFFNClassifier  #: define classifier class
    head_box_post_cls: DETRBoxPost = MaxFGBoxPost  #: define postprocessing strategy during inference
    matcher_class_criterion_cls: ClassCriterion = SimpleClassCriterionSoftmax  #: criterion to compute class cost matrix
