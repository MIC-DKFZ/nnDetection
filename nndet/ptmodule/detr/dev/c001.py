# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Optional, Sequence, Type

from nndet.core.boxes.criterions.base import BoxCriterion, ClassCriterion
from nndet.core.boxes.criterions.box import GIoUCenterBoxCriterion, L1RegCriterion
from nndet.core.boxes.criterions.cls import (
    FocalClassCriterionSigmoid,
    SimpleClassCriterionSigmoid,
    SimpleClassCriterionSoftmax,
)
from nndet.core.boxes.matcher1to1.base import BaseMatcher
from nndet.core.boxes.matcher1to1.hungarian import HungarianMatcher
from nndet.core.post.detr import DETRBoxPost, MaxFGBoxPost, TopKBoxPost
from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.backbone.blueprints.conv import ConvBackbone
from nndet.nn.backbone.blueprints.resconv import ResConvBackbone
from nndet.nn.heads.classifier.ffn import (
    BCEFFNClassifier,
    CEFFNClassifier,
    FFNClassifier,
    FocalFFNClassifier,
)
from nndet.nn.heads.detr.base import DETRHead
from nndet.nn.heads.detr.cdetr import ConditionalDETRHead
from nndet.nn.heads.regressor.ffn import FFNRegressor, L1GIoUFFNRegressor
from nndet.nn.layers.conv import ConvInstanceRelu
from nndet.nn.layers.conv.conv_only import ConvOnly
from nndet.nn.layers.linear import LayerLinearReluDrop
from nndet.nn.layers.pos_embed.base import BasePositionEmbedding
from nndet.nn.layers.pos_embed.sine import PositionEmbeddingSine
from nndet.nn.neck.channel_mapper import ChannelMapper
from nndet.nn.transformer.abstract_transformer import AbstractTransformer
from nndet.nn.transformer.detr_transformer import DETRTransformer
from nndet.nn.transformer.layers.abstract import (
    BaseTransformerDecoder,
    BaseTransformerEncoder,
)
from nndet.nn.transformer.layers.conditional_detr import (
    ConditionalDETRTransformerDecoder,
)
from nndet.nn.transformer.layers.detr import (
    DETRTransformerDecoder,
    DETRTransformerEncoder,
)
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.detr.boxdetr import BoxDETRModule
from nndet.utils.typing import CONVSEQ, LINEARSEQ


@MODULE_REGISTRY.register
class BoxDETRC001(BoxDETRModule):
    backbone_cls: Type[AbstractBackbone] = ConvBackbone  #: define class for backbone
    backbone_conv_cls: Type[CONVSEQ] = ConvInstanceRelu  #: conv class used for backbone
    channel_mapper_cls: Type[ChannelMapper] = ChannelMapper
    channel_mapper_conv_cls: Type[CONVSEQ] = ConvOnly
    # transformer
    pos_embed_cls: BasePositionEmbedding = PositionEmbeddingSine
    transformer_encoder_cls: BaseTransformerEncoder = DETRTransformerEncoder
    transformer_decoder_cls: BaseTransformerDecoder = DETRTransformerDecoder
    transformer_cls: AbstractTransformer = DETRTransformer

    # head blocks
    head_cls: DETRHead = DETRHead  #: main DETR head
    head_linear_cls: LINEARSEQ = LayerLinearReluDrop  #: conv class used for head
    head_classifier_cls: FFNClassifier = CEFFNClassifier  #: define classifier class
    head_regressor_cls: FFNRegressor = L1GIoUFFNRegressor  #: define regressor class
    head_box_post_cls: DETRBoxPost = MaxFGBoxPost  #: define postprocessing strategy during inference

    matcher_cls: BaseMatcher = HungarianMatcher  #: matching algorithm
    matcher_class_criterion_cls: ClassCriterion = SimpleClassCriterionSoftmax  #: criterion to compute class cost matrix
    # either reg or box criterion need to be set
    # reg criterion usually operates on encoded targets while box cirterion operates on raw boxes
    # there is no structural difference though and just a nomenclature
    matcher_reg_criterion_cls: Optional[BoxCriterion] = L1RegCriterion  #: criterion to compute regression cost matrix
    matcher_box_criterion_cls: Optional[
        BoxCriterion
    ] = GIoUCenterBoxCriterion  #: criterion to compute regression cost matrix


@MODULE_REGISTRY.register
class BoxDETRC001CE(BoxDETRC001):
    pass


@MODULE_REGISTRY.register
class BoxDETRC001CE_S16(BoxDETRC001CE):
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
class BoxDETRC001CERes(BoxDETRC001):
    backbone_cls: Type[AbstractBackbone] = ResConvBackbone  #: define class for backbone


@MODULE_REGISTRY.register
class BoxDETRC001BCE(BoxDETRC001):
    head_classifier_cls: FFNClassifier = BCEFFNClassifier  #: define classifier class
    head_box_post_cls: DETRBoxPost = TopKBoxPost  #: define postprocessing strategy during inference
    matcher_class_criterion_cls: ClassCriterion = SimpleClassCriterionSigmoid  #: criterion to compute class cost matrix


@MODULE_REGISTRY.register
class BoxDETRC001BCERes(BoxDETRC001):
    backbone_cls: Type[AbstractBackbone] = ResConvBackbone  #: define class for backbone
    head_classifier_cls: FFNClassifier = BCEFFNClassifier  #: define classifier class
    head_box_post_cls: DETRBoxPost = TopKBoxPost  #: define postprocessing strategy during inference
    matcher_class_criterion_cls: ClassCriterion = SimpleClassCriterionSigmoid  #: criterion to compute class cost matrix


@MODULE_REGISTRY.register
class BoxDETRC001Focal(BoxDETRC001):
    head_classifier_cls: FFNClassifier = FocalFFNClassifier  #: define classifier class
    head_box_post_cls: DETRBoxPost = TopKBoxPost  #: define postprocessing strategy during inference
    matcher_class_criterion_cls: ClassCriterion = FocalClassCriterionSigmoid  #: criterion to compute class cost matrix


@MODULE_REGISTRY.register
class BoxDETRC001Focal_S16(BoxDETRC001Focal):
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
class BoxDETRC001FocalRes(BoxDETRC001):
    backbone_cls: Type[AbstractBackbone] = ResConvBackbone  #: define class for backbone
    head_classifier_cls: FFNClassifier = FocalFFNClassifier  #: define classifier class
    head_box_post_cls: DETRBoxPost = TopKBoxPost  #: define postprocessing strategy during inference
    matcher_class_criterion_cls: ClassCriterion = FocalClassCriterionSigmoid  #: criterion to compute class cost matrix


@MODULE_REGISTRY.register
class BoxCDETRC001Focal(BoxDETRC001):
    transformer_encoder_cls: BaseTransformerEncoder = DETRTransformerEncoder
    transformer_decoder_cls: BaseTransformerDecoder = ConditionalDETRTransformerDecoder
    transformer_cls: AbstractTransformer = DETRTransformer
    backbone_cls: Type[AbstractBackbone] = ConvBackbone  #: define class for backbone

    head_cls: DETRHead = ConditionalDETRHead  #: main DETR head
    head_classifier_cls: FFNClassifier = FocalFFNClassifier  #: define classifier class
    head_box_post_cls: DETRBoxPost = TopKBoxPost  #: define postprocessing strategy during inference
    matcher_class_criterion_cls: ClassCriterion = FocalClassCriterionSigmoid  #: criterion to compute class cost matrix


@MODULE_REGISTRY.register
class BoxCDETRC001Focal_S16(BoxCDETRC001Focal):
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
class BoxCDETRC001ResFocal(BoxDETRC001):
    transformer_encoder_cls = DETRTransformerEncoder
    transformer_decoder_cls = ConditionalDETRTransformerDecoder
    transformer_cls = DETRTransformer
    backbone_cls: Type[AbstractBackbone] = ResConvBackbone  #: define class for backbone

    head_cls: DETRHead = ConditionalDETRHead  #: main DETR head
    head_classifier_cls: FFNClassifier = FocalFFNClassifier  #: define classifier class
    head_box_post_cls: DETRBoxPost = TopKBoxPost  #: define postprocessing strategy during inference
    matcher_class_criterion_cls: ClassCriterion = FocalClassCriterionSigmoid  #: criterion to compute class cost matrix


@MODULE_REGISTRY.register
class BoxIOCDETRC001Focal_S16(BoxCDETRC001Focal):
    @classmethod
    def use_box_io(cls):
        return True
