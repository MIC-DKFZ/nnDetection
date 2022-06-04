# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Type

from nndet.arch.blocks import StackedConvBlock2
from nndet.arch.blocks.basic import AbstractBlock
from nndet.arch.conv import ConvGroupLReLU, ConvGroupMish, ConvInstanceLReLU
from nndet.arch.decoder.base import BaseUFPN, UFPNModular
from nndet.arch.encoder import Encoder
from nndet.arch.encoder.abstract import AbstractEncoder
from nndet.arch.heads.classifier.dense import BCECLassifier, DenseClassifier
from nndet.arch.heads.classifier.roi import BCEConvRoIClassifier, RoIClassifier
from nndet.arch.heads.comb import BoxHeadHNM
from nndet.arch.heads.comb.base import AnchorHead
from nndet.arch.heads.comb.roi import RoIBoxHead
from nndet.arch.heads.masker.base import BCESingleMasker, Masker
from nndet.arch.heads.regressor.dense import DenseRegressor, L1Regressor
from nndet.arch.heads.regressor.roi import L1ConvRoIRegressor, RoIRegressor
from nndet.arch.heads.segmenter import DiCESegmenterFgBg, Segmenter
from nndet.core.abstract import AbstractDetector, AbstractOneStageDetector
from nndet.core.boxes.matcher import ATSSMatcher, IoUMatcher, Matcher
from nndet.core.boxes.sampler import HardNegativeSamplerBatched, SamplerType
from nndet.core.post.box import BoxPostprocessing, CrossLevelBoxPostprocessing
from nndet.core.post.mask import MaskPostprocessing, NoMaskPostprocessing
from nndet.core.rcnn import RCNN
from nndet.core.retina import BaseRetinaNet
from nndet.core.rois.module import RoIModule
from nndet.core.rois.pooler import RoIAlignNaiveAssign, RoIPooler
from nndet.inference.ensembler.base import BaseEnsembler
from nndet.inference.ensembler.detection import BoxEnsemblerSelective
from nndet.inference.ensembler.mask import MaskViaBoxesSelectiveEnsembler
from nndet.inference.sweeper import BoxSweeper, MaskSweeper, Sweeper
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.mixins.evaluation import BoxWithRPNEvalMixin, ScoreMasksEvalMixin
from nndet.ptmodule.mixins.model import TwoStageMixin
from nndet.ptmodule.mixins.prediction import MaskViaBoxPredictionMixin
from nndet.ptmodule.mixins.prepare import (
    BinaryMasksPrepareMixin,
    BoxesPrepareMixin,
    SemanticFgPrepareMixin,
)
from nndet.ptmodule.module import LightningBaseModule
from nndet.utils.typing import CONVSEQ


@MODULE_REGISTRY.register
class MaskURCNNC002(
    LightningBaseModule,  # Detection Base
    BinaryMasksPrepareMixin,  # prepare binary masks for instance segmentation training
    SemanticFgPrepareMixin,  # prepare batch for semantic segmentation training
    BoxesPrepareMixin,  # prepare batch for box training
    BoxWithRPNEvalMixin,  # Bounding Box Evaluation (with RPN)
    TwoStageMixin,  # Single Stage Detector
    MaskViaBoxPredictionMixin,  # Mask Sweep
    ScoreMasksEvalMixin,  # Mask Evaluations
):
    full_detector_cls: Type[AbstractDetector] = RCNN  # Two stage detector class RCNN
    # Use `detector_cls` to set RPN module class
    # define RPN cls
    detector_cls: Type[AbstractOneStageDetector] = BaseRetinaNet

    ###################
    # RPN Configuration
    ###################
    backbone_cls: Type[AbstractEncoder] = Encoder  # define class for backbone
    backbone_conv_cls: Type[CONVSEQ] = ConvInstanceLReLU  # conv class used for backbone
    backbone_block: Type[
        AbstractBlock
    ] = StackedConvBlock2  # define central building block of backbone

    neck_cls: Type[BaseUFPN] = UFPNModular  # define class for neck
    neck_conv_cls: Type[CONVSEQ] = ConvInstanceLReLU  # conv class used for neck

    head_cls: Type[AnchorHead] = BoxHeadHNM  # define class for head
    head_conv_cls: Type[CONVSEQ] = ConvGroupLReLU  # conv class used for head
    head_classifier_cls: Type[
        DenseClassifier
    ] = BCECLassifier  # define class for head classifier
    head_regressor_cls: Type[
        DenseRegressor
    ] = L1Regressor  # define class for head regressor
    # [optional] sampler class for negative mining
    # if None: no sampler will be given to the head
    head_sampler_cls: Optional[Type[SamplerType]] = HardNegativeSamplerBatched

    matcher_cls: Type[
        Matcher
    ] = ATSSMatcher  # define class to match anchors to ground truth
    segmenter_cls: Optional[
        Type[Segmenter]
    ] = DiCESegmenterFgBg  # segmentation head as in RetinaUNet

    ########################
    # RoI Head Configuration
    ########################
    # RoI classes
    roi_conv_cls: Type[CONVSEQ] = ConvGroupLReLU  # conv class used for RoI head
    roi_module_cls: Type[RoIModule] = RoIModule  # class of RoI module
    roi_head_cls: Type[RoIBoxHead] = RoIBoxHead  # class of box head of RoI module
    roi_classifier_cls: Type[
        RoIClassifier
    ] = BCEConvRoIClassifier  # box head classifier class
    roi_regressor_cls: Type[
        RoIRegressor
    ] = L1ConvRoIRegressor  # box head regressor class

    roi_matcher_cls: Type[Matcher] = IoUMatcher  # class of RoI matcher
    roi_sampler_cls: Type[
        SamplerType
    ] = HardNegativeSamplerBatched  # class of RoI sampler
    roi_box_pooler_cls: Type[RoIPooler] = RoIAlignNaiveAssign  # class of RoI box pooler
    roi_box_post_cls: Type[
        BoxPostprocessing
    ] = CrossLevelBoxPostprocessing  # define roi box postprocessing strategy

    roi_masker_cls: Type[Masker] = BCESingleMasker  # class of RoI mask head
    roi_mask_pooler_cls: Type[
        RoIPooler
    ] = RoIAlignNaiveAssign  # class of RoI mask pooler
    roi_mask_post_cls: Type[
        MaskPostprocessing
    ] = NoMaskPostprocessing  # define roi mask postprocessing strategy

    @classmethod
    def get_ensembler_cls(cls, dim: int) -> Type[BaseEnsembler]:
        """
        Returns:
            Type[BaseEnsembler]: return class of ensembler to use for this
                class
        """
        return BoxEnsemblerSelective

    @classmethod
    def get_sweeper_cls(cls) -> Type[Sweeper]:
        return BoxSweeper


@MODULE_REGISTRY.register
class MaskURCNNC002MishRoI(MaskURCNNC002):
    roi_conv_cls: Type[CONVSEQ] = ConvGroupMish  # conv class used for RoI head


@MODULE_REGISTRY.register
class MaskURCNNC002FullMask(MaskURCNNC002):
    @classmethod
    def get_ensembler_cls(cls, dim: int) -> Type[BaseEnsembler]:
        """
        Returns:
            Type[BaseEnsembler]: return class of ensembler to use for this
                class
        """
        return MaskViaBoxesSelectiveEnsembler

    @classmethod
    def get_sweeper_cls(cls) -> Type[Sweeper]:
        """
        Returns:
            Type[Sweeper]: return class of sweeper to use for this class
        """
        return MaskSweeper
