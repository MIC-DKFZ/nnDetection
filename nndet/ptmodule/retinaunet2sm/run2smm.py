# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Type

from nndet.core.abstract import AbstractDetector, AbstractOneStageDetector
from nndet.core.boxes.matcher import Matcher
from nndet.core.boxes.sampler import AbstractSampler
from nndet.core.post.box import BoxPostprocessing
from nndet.core.post.mask import MaskPostprocessing, NoMaskPostprocessing
from nndet.core.rcnn import RCNN
from nndet.core.rois.module import RoIModule
from nndet.core.rois.pooler import RoIPooler
from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.heads.classifier.dense import DenseClassifier
from nndet.nn.heads.classifier.roi import RoIClassifier
from nndet.nn.heads.comb.base import AnchorHead
from nndet.nn.heads.comb.roi import RoIBoxHead
from nndet.nn.heads.masker.roi import Masker
from nndet.nn.heads.regressor.dense import DenseRegressor
from nndet.nn.heads.regressor.roi import RoIRegressor
from nndet.nn.heads.segmenter import Segmenter
from nndet.nn.neck.abstract import AbstractNeck
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.mixins.evaluation import BoxWithRPNEvalMixin
from nndet.ptmodule.mixins.model import TwoStageMixin
from nndet.ptmodule.mixins.prediction.boxes import BoxPredictionMixinV2
from nndet.ptmodule.mixins.prepare import (
    BinaryMasksPrepareMixin,
    BoxesPrepareMixin,
    SemanticFgPrepareMixin,
)
from nndet.ptmodule.module import LightningBaseModule
from nndet.utils.typing import CONVSEQ


@MODULE_REGISTRY.register
class RetinaUNet2SMModule(
    LightningBaseModule,  # Detection Base
    BinaryMasksPrepareMixin,  # prepare binary masks for instance segmentation training
    SemanticFgPrepareMixin,  # prepare batch for semantic segmentation training
    BoxesPrepareMixin,  # prepare batch for box training
    BoxWithRPNEvalMixin,  # Bounding Box Evaluation (with RPN)
    TwoStageMixin,  # Single Stage Detector
    BoxPredictionMixinV2,  # Bounding Box Sweep
):
    """
    MaskRCNNModule with Box Output
    """

    full_detector_cls: Type[AbstractDetector] = RCNN  # Two stage detector class RCNN
    # Use `detector_cls` to set RPN module class
    # define RPN cls
    detector_cls: Type[AbstractOneStageDetector] = ...

    ###################
    # RPN Configuration
    ###################
    backbone_cls: Type[AbstractBackbone] = ...  # define class for backbone
    backbone_conv_cls: Type[CONVSEQ] = ...  # conv class used for backbone

    neck_cls: Type[AbstractNeck] = ...  # define class for neck
    neck_conv_cls: Type[CONVSEQ] = ...  # conv class used for neck

    head_cls: Type[AnchorHead] = ...  # define class for head
    head_conv_cls: Type[CONVSEQ] = ...  # conv class used for head
    head_classifier_cls: Type[DenseClassifier] = ...  # define class for head classifier
    head_regressor_cls: Type[DenseRegressor] = ...  # define class for head regressor
    # [optional] sampler class for negative mining
    # if None: no sampler will be given to the head
    head_sampler_cls: Optional[Type[AbstractSampler]] = ...

    matcher_cls: Type[Matcher] = ...  # define class to match anchors to ground truth
    box_post_cls: Type[BoxPostprocessing] = ...  # define box postprocessing strategy
    segmenter_cls: Optional[Type[Segmenter]] = ...  # segmentation head as in RetinaUNet

    ########################
    # RoI Head Configuration
    ########################
    # RoI classes
    roi_conv_cls: Type[CONVSEQ] = ...  # conv class used for RoI head
    roi_module_cls: Type[RoIModule] = ...  # class of RoI module
    roi_head_cls: Type[RoIBoxHead] = ...  # class of box head of RoI module
    roi_classifier_cls: Type[RoIClassifier] = ...  # box head classifier class
    roi_regressor_cls: Type[RoIRegressor] = ...  # box head regressor class

    roi_matcher_cls: Type[Matcher] = ...  # class of RoI matcher
    roi_sampler_cls: Type[AbstractSampler] = ...  # class of RoI sampler
    roi_box_pooler_cls: Type[RoIPooler] = ...  # class of RoI box pooler
    roi_box_post_cls: Type[BoxPostprocessing] = ...  # define roi box postprocessing strategy

    roi_masker_cls: Type[Masker] = ...  # class of RoI mask head
    roi_mask_pooler_cls: Type[RoIPooler] = ...  # class of RoI mask pooler
    roi_mask_post_cls: Type[MaskPostprocessing] = NoMaskPostprocessing  # define roi mask postprocessing strategy


@MODULE_REGISTRY.register
class RetinaNet2SMModule(
    LightningBaseModule,  # Detection Base
    BinaryMasksPrepareMixin,  # prepare binary masks for instance segmentation training
    BoxesPrepareMixin,  # prepare batch for box training
    BoxWithRPNEvalMixin,  # Bounding Box Evaluation (with RPN)
    TwoStageMixin,  # Single Stage Detector
    BoxPredictionMixinV2,  # Bounding Box Sweep
):
    """
    MaskRCNNModule with Box Output
    """

    full_detector_cls: Type[AbstractDetector] = RCNN  # Two stage detector class RCNN
    # Use `detector_cls` to set RPN module class
    # define RPN cls
    detector_cls: Type[AbstractOneStageDetector] = ...

    ###################
    # RPN Configuration
    ###################
    backbone_cls: Type[AbstractBackbone] = ...  # define class for backbone
    backbone_conv_cls: Type[CONVSEQ] = ...  # conv class used for backbone

    neck_cls: Type[AbstractNeck] = ...  # define class for neck
    neck_conv_cls: Type[CONVSEQ] = ...  # conv class used for neck

    head_cls: Type[AnchorHead] = ...  # define class for head
    head_conv_cls: Type[CONVSEQ] = ...  # conv class used for head
    head_classifier_cls: Type[DenseClassifier] = ...  # define class for head classifier
    head_regressor_cls: Type[DenseRegressor] = ...  # define class for head regressor
    # [optional] sampler class for negative mining
    # if None: no sampler will be given to the head
    head_sampler_cls: Optional[Type[AbstractSampler]] = ...

    matcher_cls: Type[Matcher] = ...  # define class to match anchors to ground truth
    box_post_cls: Type[BoxPostprocessing] = ...  # define box postprocessing strategy
    # Use `MaskURCNNModule` for configurations where `segmenter_cls` is not None!
    # Classes other than None are not supprted here
    segmenter_cls: Optional[Type[Segmenter]] = None  # [optional] segmentation head as in RetinaUNet

    ########################
    # RoI Head Configuration
    ########################
    # RoI classes
    roi_conv_cls: Type[CONVSEQ] = ...  # conv class used for RoI head
    roi_module_cls: Type[RoIModule] = ...  # class of RoI module
    roi_head_cls: Type[RoIBoxHead] = ...  # class of box head of RoI module
    roi_classifier_cls: Type[RoIClassifier] = ...  # box head classifier class
    roi_regressor_cls: Type[RoIRegressor] = ...  # box head regressor class

    roi_matcher_cls: Type[Matcher] = ...  # class of RoI matcher
    roi_sampler_cls: Type[AbstractSampler] = ...  # class of RoI sampler
    roi_box_pooler_cls: Type[RoIPooler] = ...  # class of RoI box pooler
    roi_box_post_cls: Type[BoxPostprocessing] = ...  # define roi box postprocessing strategy

    roi_masker_cls: Type[Masker] = ...  # class of RoI mask head
    roi_mask_pooler_cls: Type[RoIPooler] = ...  # class of RoI mask pooler
    roi_mask_post_cls: Type[MaskPostprocessing] = NoMaskPostprocessing  # define roi mask postprocessing strategy
