# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Optional, Sequence, Type

from nndet.core.boxes.criterions.base import BoxCriterion, ClassCriterion
from nndet.core.boxes.criterions.box import GIoUCenterBoxCriterion, L1RegCriterion
from nndet.core.boxes.criterions.cls import FocalClassCriterionSigmoid
from nndet.core.boxes.matcher1to1.base import BaseMatcher
from nndet.core.boxes.matcher1to1.hungarian import HungarianMatcher
from nndet.core.post.detr import DETRBoxPost, TopKBoxPost
from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.backbone.blueprints.conv import ConvBackbone
from nndet.nn.heads.classifier.ffn import FFNClassifier, FocalFFNClassifier
from nndet.nn.heads.detr.base import DETRHead
from nndet.nn.heads.detr.cdetr import ConditionalDETRHead
from nndet.nn.heads.detr.deformable_detr import DeformableDETRHead
from nndet.nn.heads.regressor.ffn import FFNRegressor, L1UGIoUFFNRegressor
from nndet.nn.layers.conv import ConvInstanceRelu
from nndet.nn.layers.conv.conv_only import ConvOnly
from nndet.nn.layers.conv.group import ConvGroupRelu
from nndet.nn.layers.linear import LayerLinearReluDrop
from nndet.nn.layers.pos_embed.base import BasePositionEmbedding
from nndet.nn.layers.pos_embed.sine import PositionEmbeddingSine
from nndet.nn.neck.channel_mapper import ChannelMapper
from nndet.nn.transformer.abstract_transformer import AbstractTransformer
from nndet.nn.transformer.deformable_transformer import DeformableDETRTransformer
from nndet.nn.transformer.detr_transformer import DETRTransformer
from nndet.nn.transformer.layers.abstract import (
    BaseTransformerDecoder,
    BaseTransformerEncoder,
)
from nndet.nn.transformer.layers.conditional_detr import (
    ConditionalDETRTransformerDecoder,
)
from nndet.nn.transformer.layers.deformable_detr import (
    DeformableDETRTransformerDecoder,
    DeformableDETRTransformerEncoder,
)
from nndet.nn.transformer.layers.detr import (
    DETRTransformerDecoder,
    DETRTransformerEncoder,
)
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.mixins.evaluation.boxes import BoxEvalMixin
from nndet.ptmodule.mixins.model.set import DeformableSetModelMixin, DETRModelMixin
from nndet.ptmodule.mixins.prediction.boxes import BoxPredictionMixinV2
from nndet.ptmodule.mixins.prepare.boxes import BoxesPrepareMixin
from nndet.ptmodule.module import LightningBaseModule
from nndet.utils.typing import CONVSEQ, LINEARSEQ


@MODULE_REGISTRY.register
class BoxDETRV002(
    LightningBaseModule,  # Main module
    BoxesPrepareMixin,  # prepare batch for box training
    BoxEvalMixin,  # Bounding Box Evaluation
    DETRModelMixin,  # DETR Mixin to build the model
    BoxPredictionMixinV2,  # Bounding Box Sweep
):
    backbone_cls: Type[AbstractBackbone] = ConvBackbone  #: define class for backbone
    backbone_conv_cls: Type[CONVSEQ] = ConvInstanceRelu  #: conv class used for backbone
    channel_mapper_cls: Type[ChannelMapper] = ChannelMapper  #: map channels from backbone to transformer
    channel_mapper_conv_cls: Type[CONVSEQ] = ConvOnly  #: conv class used for channel mapper

    # transformer
    transformer_cls: Type[AbstractTransformer] = DETRTransformer  #: define detector transformer architecture
    pos_embed_cls: BasePositionEmbedding = PositionEmbeddingSine  #: define positional embedding for feature maps
    transformer_encoder_cls: BaseTransformerEncoder = DETRTransformerEncoder  #: define encoder class of transformer
    transformer_decoder_cls: BaseTransformerDecoder = DETRTransformerDecoder  #: define decoder class of transformer

    # head blocks
    head_cls: DETRHead = DETRHead  #: main DETR head
    head_linear_cls: LINEARSEQ = LayerLinearReluDrop  #: conv class used for head
    head_classifier_cls: FFNClassifier = FocalFFNClassifier  #: define classifier class
    head_regressor_cls: FFNRegressor = L1UGIoUFFNRegressor  #: define regressor class
    head_box_post_cls: DETRBoxPost = TopKBoxPost  #: define postprocessing strategy during inference

    matcher_cls: BaseMatcher = HungarianMatcher  #: matching algorithm
    matcher_class_criterion_cls: ClassCriterion = FocalClassCriterionSigmoid  #: criterion to compute class cost matrix
    # either reg or box criterion need to be set
    # reg criterion usually operates on encoded targets while box cirterion operates on raw boxes
    # there is no structural difference though and just a nomenclature
    matcher_reg_criterion_cls: Optional[BoxCriterion] = L1RegCriterion  #: criterion to compute regression cost matrix
    matcher_box_criterion_cls: Optional[
        BoxCriterion
    ] = GIoUCenterBoxCriterion  #: criterion to compute regression cost matrix

    @classmethod
    def _build_backbone(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        patch_size: Optional[Sequence[int]] = None,
    ) -> AbstractBackbone:
        _plan_arch = copy.deepcopy(plan_arch)
        _plan_arch["conv_kernels"] = _plan_arch["conv_kernels"][:-1]
        _plan_arch["strides"] = _plan_arch["strides"][:-1]
        return super()._build_backbone(
            plan_arch=_plan_arch,
            model_cfg=model_cfg,
            patch_size=patch_size,
        )


@MODULE_REGISTRY.register
class BoxCDETRV002(BoxDETRV002):
    # transformer
    transformer_decoder_cls: BaseTransformerDecoder = (
        ConditionalDETRTransformerDecoder  #: define decoder class of transformer
    )

    # head blocks
    head_cls: DETRHead = ConditionalDETRHead  #: main DETR head


@MODULE_REGISTRY.register
class BoxDeformableDETRV002(
    LightningBaseModule,  # Main module
    BoxesPrepareMixin,  # prepare batch for box training
    BoxEvalMixin,  # Bounding Box Evaluation
    DeformableSetModelMixin,  # DETR Mixin to build the model
    BoxPredictionMixinV2,  # Bounding Box Sweep
):
    backbone_cls: Type[AbstractBackbone] = ConvBackbone  #: define class for backbone
    backbone_conv_cls: Type[CONVSEQ] = ConvInstanceRelu  #: conv class used for backbone
    channel_mapper_cls: Type[ChannelMapper] = ChannelMapper  #: map channels from backbone to transformer
    channel_mapper_conv_cls: Type[CONVSEQ] = ConvGroupRelu  #: conv class used for channel mapper

    # transformer
    transformer_cls: Type[AbstractTransformer] = DeformableDETRTransformer  #: define detector transformer architecture
    pos_embed_cls: BasePositionEmbedding = PositionEmbeddingSine  #: define positional embedding for feature maps
    transformer_encoder_cls: BaseTransformerEncoder = (
        DeformableDETRTransformerEncoder  #: define encoder class of transformer
    )
    transformer_decoder_cls: BaseTransformerDecoder = (
        DeformableDETRTransformerDecoder  #: define decoder class of transformer
    )

    # head blocks
    head_cls: DETRHead = DeformableDETRHead  #: main DETR head
    head_linear_cls: LINEARSEQ = LayerLinearReluDrop  #: conv class used for head
    head_classifier_cls: FFNClassifier = FocalFFNClassifier  #: define classifier class
    head_regressor_cls: FFNRegressor = L1UGIoUFFNRegressor  #: define regressor class
    head_box_post_cls: DETRBoxPost = TopKBoxPost  #: define postprocessing strategy during inference

    matcher_cls: BaseMatcher = HungarianMatcher  #: matching algorithm
    matcher_class_criterion_cls: ClassCriterion = FocalClassCriterionSigmoid  #: criterion to compute class cost matrix
    # either reg or box criterion need to be set
    # reg criterion usually operates on encoded targets while box cirterion operates on raw boxes
    # there is no structural difference though and just a nomenclature
    matcher_reg_criterion_cls: Optional[BoxCriterion] = L1RegCriterion  #: criterion to compute regression cost matrix
    matcher_box_criterion_cls: Optional[
        BoxCriterion
    ] = GIoUCenterBoxCriterion  #: criterion to compute regression cost matrix
