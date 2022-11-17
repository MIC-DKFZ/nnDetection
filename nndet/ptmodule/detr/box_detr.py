from typing import Optional, Type

from nndet.core.abstract import AbstractOneStageDetector
from nndet.core.boxes.criterions.base import BoxCriterion, ClassCriterion
from nndet.core.boxes.criterions.box import GIoUBoxCriterion, L1RegCriterion
from nndet.core.boxes.criterions.cls import (
    SimpleClassCriterionSigmoid,
    SimpleClassCriterionSoftmax,
)
from nndet.core.boxes.matcher1to1.base import BaseMatcher
from nndet.core.boxes.matcher1to1.hungarian import HungarianMatcher
from nndet.core.detr import BaseDETR
from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.backbone.blueprints.conv import ConvBackbone
from nndet.nn.backbone.blueprints.resconv import ResConvBackbone
from nndet.nn.heads.classifier.ffn import (
    CEFFNClassifier,
    FFNClassifier,
    FocalFFNClassifier,
)
from nndet.nn.heads.detr.base import DETRHead
from nndet.nn.heads.regressor.ffn import FFNRegressor, L1GIoUFFNRegressor
from nndet.nn.heads.segmenter import Segmenter
from nndet.nn.layers.conv import ConvInstanceRelu
from nndet.nn.layers.linear import LayerLinearReluDrop
from nndet.nn.layers.pos_embed.base import BasePositionEmbedding
from nndet.nn.layers.pos_embed.sine import PositionEmbeddingSine
from nndet.nn.neck.abstract import AbstractNeck
from nndet.nn.transformer import TransformerFacebook
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.mixins.evaluation import BoxEvalMixin
from nndet.ptmodule.mixins.model.detr import SetModelMixin
from nndet.ptmodule.mixins.prediction import BoxPredictionMixin
from nndet.ptmodule.mixins.prepare import BoxesPrepareMixin

# from nndet.ptmodule.mixins.train import TrainMixin
from nndet.ptmodule.module import LightningBaseModule
from nndet.utils.typing import CONVSEQ, LINEARSEQ


class BoxDETRModule(
    # TrainMixin,
    LightningBaseModule,  # Main module
    BoxesPrepareMixin,  # prepare batch for box training
    BoxEvalMixin,  # Bounding Box Evaluation
    SetModelMixin,  # DETR Mixin to build the model
    BoxPredictionMixin,  # Bounding Box Sweep
):
    # define detector cls
    detector_cls: Type[AbstractOneStageDetector] = BaseDETR  #: define base detector class

    backbone_cls: Type[AbstractBackbone] = ...  #: define class for backbone
    backbone_conv_cls: Type[CONVSEQ] = ...  #: conv class used for backbone
    # transformer
    pos_embed_cls: BasePositionEmbedding = ...
    transformer_cls = ...

    # head blocks
    head_cls: DETRHead = ...  #: main DETR head
    head_linear_cls: LINEARSEQ = ...  #: conv class used for head
    head_classifier_cls: FFNClassifier = ...  #: define classifier class
    head_regressor_cls: FFNRegressor = ...  #: define regressor class

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


@MODULE_REGISTRY.register
class BoxDETR(BoxDETRModule):
    backbone_cls: Type[AbstractBackbone] = ConvBackbone  #: define class for backbone
    backbone_conv_cls: Type[CONVSEQ] = ConvInstanceRelu  #: conv class used for backbone
    # transformer
    pos_embed_cls: BasePositionEmbedding = PositionEmbeddingSine
    transformer_cls = TransformerFacebook

    # head blocks
    head_cls: DETRHead = DETRHead  #: main DETR head
    head_linear_cls: LINEARSEQ = LayerLinearReluDrop  #: conv class used for head
    head_classifier_cls: FFNClassifier = CEFFNClassifier  #: define classifier class
    head_regressor_cls: FFNRegressor = L1GIoUFFNRegressor  #: define regressor class

    matcher_cls: BaseMatcher = HungarianMatcher  #: matching algorithm
    matcher_class_criterion_cls: ClassCriterion = SimpleClassCriterionSoftmax  #: criterion to compute class cost matrix
    # either reg or box criterion need to be set
    # reg criterion usually operates on encoded targets while box cirterion operates on raw boxes
    # there is no structural difference though and just a nomenclature
    matcher_reg_criterion_cls: Optional[BoxCriterion] = L1RegCriterion  #: criterion to compute regression cost matrix
    matcher_box_criterion_cls: Optional[BoxCriterion] = GIoUBoxCriterion  #: criterion to compute regression cost matrix


@MODULE_REGISTRY.register
class BoxDETRC001(BoxDETR):
    backbone_cls: Type[AbstractBackbone] = ResConvBackbone  #: define class for backbone
    head_classifier_cls: FFNClassifier = FocalFFNClassifier  #: define classifier class
    matcher_class_criterion_cls: ClassCriterion = SimpleClassCriterionSigmoid  #: criterion to compute class cost matrix


# @MODULE_REGISTRY.register
# class BoxUDETR(
#     TrainMixin,
#     LightningBaseModule,  # Main module
#     SemanticFgPrepareMixin,  # prepare batch for semantic segmentation training
#     BoxesPrepareMixin,  # prepare batch for box training
#     SemanticFgEvalMixin,  # Semantic Segmentation Evaluation
#     BoxEvalMixin,  # Bounding Box Evaluation
#     DETRMixin,  # DETR Mixin to build the model
#     BoxPredictionMixin,  # Bounding Box Sweep
# ):
#     """
#     This is the basic object detection module without a segmentation head
#     """

#     # define detector cls
#     detector_cls: Type[AbstractDetector] = BaseDETR

#     backbone_cls = ConvBackbone
#     backbone_conv_cls = ConvInstanceRelu

#     # transformer
#     pos_embed_cls: BasePositionEmbedding = PositionEmbeddingSine
#     transformer_cls = TransformerFacebook

#     # Head Blocks
#     head_cls = BaseSoftmaxDETRHead

#     neck_cls: Type[AbstractNeck] = UpFPN  # define class for neck
#     neck_conv_cls: Type[CONVSEQ] = partial(
#         ConvGroupLReLU, initializer=InitHeV2(mode="fan_out")
#     )  # conv class used for neck
#     segmenter_cls = DiCESegmenterFgBg
