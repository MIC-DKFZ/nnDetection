# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import Optional, Type

from nndet.core.abstract import AbstractDetector, AbstractOneStageDetector
from nndet.core.boxes.matcher import ATSSMatcher, IoUMatcher, Matcher
from nndet.core.boxes.sampler import AbstractSampler, HardNegativeSamplerBatched
from nndet.core.post.box import (
    BoxPostprocessing,
    CrossLevelBoxPostprocessing,
    PerLevelBoxPostprocessing,
)
from nndet.core.post.mask import MaskPostprocessing, NoMaskPostprocessing
from nndet.core.rcnn import RCNN
from nndet.core.retina import BaseRetinaNet
from nndet.core.rois.module import RoIModule
from nndet.core.rois.pooler import RoIAlignNaiveAssign, RoIPooler
from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.backbone.blueprints.conv import ConvBackbone
from nndet.nn.backbone.blueprints.resconv import ResConvBackbone
from nndet.nn.heads.classifier import FocalClassifier
from nndet.nn.heads.classifier.dense import BCECLassifier, DenseClassifier
from nndet.nn.heads.classifier.roi import (
    BCEConvRoIClassifier,
    BCEFCRoIClassifier,
    RoIClassifier,
)
from nndet.nn.heads.comb import BoxHeadAll, BoxHeadHNM
from nndet.nn.heads.comb.base import AnchorHead
from nndet.nn.heads.comb.roi import RoIBoxHead
from nndet.nn.heads.masker.roi import BCEAgnosticMasker, BCESpecificMasker, Masker
from nndet.nn.heads.regressor.dense import DenseRegressor, L1Regressor
from nndet.nn.heads.regressor.roi import (
    L1ConvRoIRegressor,
    L1FCRoIRegressor,
    RoIRegressor,
)
from nndet.nn.heads.segmenter import DiCESegmenterFgBg, Segmenter
from nndet.nn.layers.conv import ConvBatchLReLU, ConvGroupLReLU, ConvInstanceLReLU
from nndet.nn.layers.initializer import InitHeV2
from nndet.nn.neck.abstract import AbstractNeck
from nndet.nn.neck.fpn import FPN, UFPN
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.mixins.evaluation import BoxWithRPNEvalMixin
from nndet.ptmodule.mixins.model import TwoStageMixin
from nndet.ptmodule.mixins.prediction import BoxPredictionMixin
from nndet.ptmodule.mixins.prepare import (
    BinaryMasksPrepareMixin,
    BoxesPrepareMixin,
    SemanticFgPrepareMixin,
)
from nndet.ptmodule.module import LightningBaseModule
from nndet.utils.typing import CONVSEQ


@MODULE_REGISTRY.register
class BoxMaskURCNNC004(
    LightningBaseModule,  # Detection Base
    BinaryMasksPrepareMixin,  # prepare binary masks for instance segmentation training
    SemanticFgPrepareMixin,  # prepare batch for semantic segmentation training
    BoxesPrepareMixin,  # prepare batch for box training
    BoxWithRPNEvalMixin,  # Bounding Box Evaluation (with RPN)
    TwoStageMixin,  # Single Stage Detector
    BoxPredictionMixin,  # Bounding Box Sweep
):
    """
    MaskRCNNModule with Box Output
    """

    full_detector_cls: Type[AbstractDetector] = RCNN  # Two stage detector class RCNN
    # Use `detector_cls` to set RPN module class
    # define RPN cls
    detector_cls: Type[AbstractOneStageDetector] = BaseRetinaNet

    ###################
    # RPN Configuration
    ###################
    backbone_cls: Type[AbstractBackbone] = ConvBackbone  # define class for backbone
    backbone_conv_cls: Type[CONVSEQ] = ConvInstanceLReLU  # conv class used for backbone

    neck_cls: Type[AbstractNeck] = UFPN  # define class for neck
    neck_conv_cls: Type[CONVSEQ] = ConvInstanceLReLU  # conv class used for neck

    head_cls: Type[AnchorHead] = BoxHeadHNM  # define class for head
    head_conv_cls: Type[CONVSEQ] = ConvGroupLReLU  # conv class used for head
    head_classifier_cls: Type[DenseClassifier] = BCECLassifier  # define class for head classifier
    head_regressor_cls: Type[DenseRegressor] = L1Regressor  # define class for head regressor
    # [optional] sampler class for negative mining
    # if None: no sampler will be given to the head
    head_sampler_cls: Optional[Type[AbstractSampler]] = HardNegativeSamplerBatched

    matcher_cls: Type[Matcher] = ATSSMatcher  # define class to match anchors to ground truth
    box_post_cls: Type[BoxPostprocessing] = CrossLevelBoxPostprocessing  # define box postprocessing strategy
    segmenter_cls: Optional[Type[Segmenter]] = DiCESegmenterFgBg  # segmentation head as in RetinaUNet

    ########################
    # RoI Head Configuration
    ########################
    # RoI classes
    roi_conv_cls: Type[CONVSEQ] = ConvGroupLReLU  # conv class used for RoI head
    roi_module_cls: Type[RoIModule] = RoIModule  # class of RoI module
    roi_head_cls: Type[RoIBoxHead] = RoIBoxHead  # class of box head of RoI module
    roi_classifier_cls: Type[RoIClassifier] = BCEConvRoIClassifier  # box head classifier class
    roi_regressor_cls: Type[RoIRegressor] = L1ConvRoIRegressor  # box head regressor class

    roi_matcher_cls: Type[Matcher] = IoUMatcher  # class of RoI matcher
    roi_sampler_cls: Type[AbstractSampler] = HardNegativeSamplerBatched  # class of RoI sampler
    roi_box_pooler_cls: Type[RoIPooler] = RoIAlignNaiveAssign  # class of RoI box pooler
    roi_box_post_cls: Type[BoxPostprocessing] = CrossLevelBoxPostprocessing  # define roi box postprocessing strategy

    roi_masker_cls: Type[Masker] = BCEAgnosticMasker  # class of RoI mask head
    roi_mask_pooler_cls: Type[RoIPooler] = RoIAlignNaiveAssign  # class of RoI mask pooler
    roi_mask_post_cls: Type[MaskPostprocessing] = NoMaskPostprocessing  # define roi mask postprocessing strategy


@MODULE_REGISTRY.register
class BoxMaskURCNNC004Focal(BoxMaskURCNNC004):
    head_cls = BoxHeadAll
    head_sampler_cls = None

    head_regressor_cls = L1Regressor
    head_classifier_cls = FocalClassifier


@MODULE_REGISTRY.register
class BoxMaskURCNNC004PerLevelPost(BoxMaskURCNNC004):
    box_post_cls: Type[BoxPostprocessing] = PerLevelBoxPostprocessing  # define box postprocessing strategy


@MODULE_REGISTRY.register
class BoxMaskURCNNC004ResEnc(BoxMaskURCNNC004):
    backbone_cls: Type[AbstractBackbone] = ResConvBackbone


@MODULE_REGISTRY.register
class BoxMaskURCNNC004ResEncHeV2(BoxMaskURCNNC004):
    backbone_cls: Type[AbstractBackbone] = ResConvBackbone
    backbone_conv_cls: Type[CONVSEQ] = partial(
        ConvInstanceLReLU, initializer=InitHeV2(mode="fan_out")
    )  # conv class used for backbone
    neck_conv_cls: Type[CONVSEQ] = partial(
        ConvGroupLReLU, initializer=InitHeV2(mode="fan_out")
    )  # conv class used for neck
    roi_conv_cls: Type[CONVSEQ] = partial(
        ConvGroupLReLU, initializer=InitHeV2(mode="fan_out")
    )  # conv class used for RoI head


@MODULE_REGISTRY.register
class BoxMaskURCNNC004HeV2(BoxMaskURCNNC004):
    backbone_cls: Type[AbstractBackbone] = ConvBackbone
    backbone_conv_cls: Type[CONVSEQ] = partial(
        ConvInstanceLReLU, initializer=InitHeV2(mode="fan_out")
    )  # conv class used for backbone
    neck_conv_cls: Type[CONVSEQ] = partial(
        ConvGroupLReLU, initializer=InitHeV2(mode="fan_out")
    )  # conv class used for neck
    roi_conv_cls: Type[CONVSEQ] = partial(
        ConvGroupLReLU, initializer=InitHeV2(mode="fan_out")
    )  # conv class used for RoI head


@MODULE_REGISTRY.register
class BoxMaskURCNNC004ResEncPerLevelPost(BoxMaskURCNNC004):
    backbone_cls: Type[AbstractBackbone] = ResConvBackbone
    box_post_cls: Type[BoxPostprocessing] = PerLevelBoxPostprocessing  # define box postprocessing strategy


@MODULE_REGISTRY.register
class BoxMaskURCNNC004ResEncFCReg(BoxMaskURCNNC004):
    # according to paper worse results with fc reg
    roi_regressor_cls: Type[RoIRegressor] = L1FCRoIRegressor  # box head regressor class


@MODULE_REGISTRY.register
class BoxMaskURCNNC004ResEncFCCls(BoxMaskURCNNC004):
    # according to paper better than conv
    roi_classifier_cls: Type[RoIClassifier] = BCEFCRoIClassifier  # box head classifier class


@MODULE_REGISTRY.register
class BoxMaskURCNNC004ResEncRoIBN(BoxMaskURCNNC004):
    roi_conv_cls: Type[CONVSEQ] = ConvBatchLReLU


@MODULE_REGISTRY.register
class BoxFasterURCNNC004ResEnc(BoxMaskURCNNC004ResEnc):
    roi_masker_cls: Type[Masker] = None  # class of RoI mask head
    roi_mask_pooler_cls: Type[RoIPooler] = None  # class of RoI mask pooler
    roi_mask_post_cls: Type[MaskPostprocessing] = None  # define roi mask postprocessing strategy


@MODULE_REGISTRY.register
class BoxMaskRCNNC004ResEnc(BoxMaskURCNNC004ResEnc):
    neck_cls: Type[AbstractNeck] = FPN  # define class for neck

    segmenter_cls: Optional[Type[Segmenter]] = None  # segmentation head as in RetinaUNet


@MODULE_REGISTRY.register
class BoxMaskURCNNC004HeV2Focal(BoxMaskURCNNC004HeV2):
    head_cls: Type[AnchorHead] = BoxHeadAll  # define class for head
    head_classifier_cls: Type[DenseClassifier] = FocalClassifier  # define class for head classifier
    # [optional] sampler class for negative mining
    head_sampler_cls: Optional[Type[AbstractSampler]] = None


@MODULE_REGISTRY.register
class BoxMaskURCNNC004MaskSpecific(BoxMaskURCNNC004):
    roi_masker_cls = BCESpecificMasker
