from typing import Optional, Type

from nndet.core.boxes.criterions.base import BoxCriterion, ClassCriterion
from nndet.core.boxes.criterions.box import GIoUCenterBoxCriterion, L1RegCriterion
from nndet.core.boxes.criterions.cls import (
    FocalClassCriterionSigmoid,
    SimpleClassCriterionSigmoid,
    SimpleClassCriterionSoftmax,
)
from nndet.core.boxes.matcher1to1.base import BaseMatcher
from nndet.core.boxes.matcher1to1.hungarian import HungarianMatcher
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
from nndet.nn.layers.linear import LayerLinearReluDrop
from nndet.nn.layers.pos_embed.base import BasePositionEmbedding
from nndet.nn.layers.pos_embed.sine import PositionEmbeddingSine
from nndet.nn.transformer import TransformerFacebook
from nndet.nn.transformer.conditional_transformer import ConditionalTransformer
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.detr.box_detr import BoxDETRModule
from nndet.utils.typing import CONVSEQ, LINEARSEQ


@MODULE_REGISTRY.register
class BoxDETRC001(BoxDETRModule):
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
    matcher_box_criterion_cls: Optional[
        BoxCriterion
    ] = GIoUCenterBoxCriterion  #: criterion to compute regression cost matrix


@MODULE_REGISTRY.register
class BoxDETRC001CE(BoxDETRC001):
    pass


@MODULE_REGISTRY.register
class BoxDETRC001CERes(BoxDETRC001):
    backbone_cls: Type[AbstractBackbone] = ResConvBackbone  #: define class for backbone


@MODULE_REGISTRY.register
class BoxDETRC001BCE(BoxDETRC001):
    head_classifier_cls: FFNClassifier = BCEFFNClassifier  #: define classifier class
    matcher_class_criterion_cls: ClassCriterion = SimpleClassCriterionSigmoid  #: criterion to compute class cost matrix


@MODULE_REGISTRY.register
class BoxDETRC001BCERes(BoxDETRC001):
    backbone_cls: Type[AbstractBackbone] = ResConvBackbone  #: define class for backbone
    head_classifier_cls: FFNClassifier = BCEFFNClassifier  #: define classifier class
    matcher_class_criterion_cls: ClassCriterion = SimpleClassCriterionSigmoid  #: criterion to compute class cost matrix


@MODULE_REGISTRY.register
class BoxDETRC001Focal(BoxDETRC001):
    head_classifier_cls: FFNClassifier = FocalFFNClassifier  #: define classifier class
    matcher_class_criterion_cls: ClassCriterion = FocalClassCriterionSigmoid  #: criterion to compute class cost matrix


@MODULE_REGISTRY.register
class BoxDETRC001FocalRes(BoxDETRC001):
    backbone_cls: Type[AbstractBackbone] = ResConvBackbone  #: define class for backbone
    head_classifier_cls: FFNClassifier = FocalFFNClassifier  #: define classifier class
    matcher_class_criterion_cls: ClassCriterion = FocalClassCriterionSigmoid  #: criterion to compute class cost matrix


@MODULE_REGISTRY.register
class BoxCDETRC001Focal(BoxDETRC001):
    transformer_cls = ConditionalTransformer
    backbone_cls: Type[AbstractBackbone] = ConvBackbone  #: define class for backbone

    head_cls: DETRHead = ConditionalDETRHead  #: main DETR head
    head_classifier_cls: FFNClassifier = FocalFFNClassifier  #: define classifier class
    matcher_class_criterion_cls: ClassCriterion = FocalClassCriterionSigmoid  #: criterion to compute class cost matrix
