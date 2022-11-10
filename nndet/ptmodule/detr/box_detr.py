from functools import partial
from typing import Type

from nndet.core.abstract import AbstractDetector
from nndet.core.detr import BaseDETR
from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.backbone.blueprints.conv import ConvBackbone
from nndet.nn.heads.detr import BaseSoftmaxDETRHead
from nndet.nn.heads.segmenter import DiCESegmenterFgBg
from nndet.nn.layers.conv import ConvGroupLReLU, ConvInstanceRelu
from nndet.nn.layers.initializer import InitHeV2
from nndet.nn.layers.pos_embed.base import BasePositionEmbedding
from nndet.nn.layers.pos_embed.sine import PositionEmbeddingSine
from nndet.nn.neck.abstract import AbstractNeck
from nndet.nn.neck.fpn import UpFPN
from nndet.nn.transformer import TransformerFacebook
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.mixins.evaluation import BoxEvalMixin, SemanticFgEvalMixin
from nndet.ptmodule.mixins.model import DETRMixin
from nndet.ptmodule.mixins.prediction import BoxPredictionMixin
from nndet.ptmodule.mixins.prepare import BoxesPrepareMixin, SemanticFgPrepareMixin
from nndet.ptmodule.mixins.train import TrainMixin
from nndet.ptmodule.module import LightningBaseModule
from nndet.utils.typing import CONVSEQ

# This file defines the Basic DETR Modules which should be inherited by all modules
# Modules can be defined by setting the different classes


class BoxDETRModule(
    TrainMixin,
    LightningBaseModule,  # Main module
    BoxesPrepareMixin,  # prepare batch for box training
    BoxEvalMixin,  # Bounding Box Evaluation
    DETRMixin,  # DETR Mixin to build the model
    BoxPredictionMixin,  # Bounding Box Sweep
):
    """
    This is the basic object detection module without a segmentation head
    """

    # define detector cls
    detector_cls: Type[AbstractDetector] = BaseDETR

    # Backbone
    backbone_cls: Type[AbstractBackbone] = ...  # define class for backbone
    backbone_conv_cls: Type[CONVSEQ] = ...  # conv class used for backbone

    # transformer
    transformer_cls = ...
    pos_embed_cls: BasePositionEmbedding = ...
    # head
    head_cls = ...  # main head


@MODULE_REGISTRY.register
class BoxDETR(BoxDETRModule):
    backbone_cls = ConvBackbone
    backbone_conv_cls = ConvInstanceRelu
    # Transformer
    pos_embed_cls: BasePositionEmbedding = PositionEmbeddingSine
    transformer_cls = TransformerFacebook
    # Head
    head_cls = BaseSoftmaxDETRHead


@MODULE_REGISTRY.register
class BoxUDETR(
    TrainMixin,
    LightningBaseModule,  # Main module
    SemanticFgPrepareMixin,  # prepare batch for semantic segmentation training
    BoxesPrepareMixin,  # prepare batch for box training
    SemanticFgEvalMixin,  # Semantic Segmentation Evaluation
    BoxEvalMixin,  # Bounding Box Evaluation
    DETRMixin,  # DETR Mixin to build the model
    BoxPredictionMixin,  # Bounding Box Sweep
):
    """
    This is the basic object detection module without a segmentation head
    """

    # define detector cls
    detector_cls: Type[AbstractDetector] = BaseDETR

    backbone_cls = ConvBackbone
    backbone_conv_cls = ConvInstanceRelu

    # transformer
    pos_embed_cls: BasePositionEmbedding = PositionEmbeddingSine
    transformer_cls = TransformerFacebook

    # Head Blocks
    head_cls = BaseSoftmaxDETRHead

    neck_cls: Type[AbstractNeck] = UpFPN  # define class for neck
    neck_conv_cls: Type[CONVSEQ] = partial(
        ConvGroupLReLU, initializer=InitHeV2(mode="fan_out")
    )  # conv class used for neck
    segmenter_cls = DiCESegmenterFgBg
