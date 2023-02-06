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
from nndet.nn.backbone.blueprints.nextconv import ConvNeXtBackbone
from nndet.nn.heads.classifier.ffn import FFNClassifier, FocalFFNClassifier
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
class BoxDETRC002(BoxDETRModule):
    # Stride 16 + Focal Loss
    backbone_cls: Type[AbstractBackbone] = ConvBackbone  #: define class for backbone
    backbone_conv_cls: Type[CONVSEQ] = ConvInstanceRelu  #: conv class used for backbone
    # transformer
    pos_embed_cls: BasePositionEmbedding = PositionEmbeddingSine
    transformer_cls = TransformerFacebook

    # head blocks
    head_cls: DETRHead = DETRHead  #: main DETR head
    head_linear_cls: LINEARSEQ = LayerLinearReluDrop  #: conv class used for head
    head_classifier_cls: FFNClassifier = FocalFFNClassifier  #: define classifier class
    head_regressor_cls: FFNRegressor = L1GIoUFFNRegressor  #: define regressor class
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

    # Stride 16
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
class BoxIODETRC002(BoxDETRC002):
    @classmethod
    def use_box_io(cls):
        return True


@MODULE_REGISTRY.register
class BoxCDETRC002(BoxDETRC002):
    transformer_cls = ConditionalTransformer
    head_cls: DETRHead = ConditionalDETRHead  #: main DETR head


@MODULE_REGISTRY.register
class BoxIOCDETRC002(BoxCDETRC002):
    @classmethod
    def use_box_io(cls):
        return True


@MODULE_REGISTRY.register
class BoxCDETRC002NeXt(BoxDETRC002):
    backbone_cls: Type[AbstractBackbone] = ConvNeXtBackbone
