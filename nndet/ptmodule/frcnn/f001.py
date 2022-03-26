from typing import Optional, Type

from nndet.arch.blocks.basic import AbstractBlock
from nndet.arch.conv import BaseConvNormAct
from nndet.arch.decoder.base import BaseUFPN
from nndet.arch.encoder.abstract import AbstractEncoder
from nndet.arch.heads.classifier.dense import DenseClassifier
from nndet.arch.heads.classifier.roi import RoIClassifier
from nndet.arch.heads.comb.base import AnchorHead
from nndet.arch.heads.comb.roi import RoIBoxHead
from nndet.arch.heads.regressor.dense import DenseRegressor
from nndet.arch.heads.regressor.roi import RoIRegressor
from nndet.arch.heads.segmenter import Segmenter
from nndet.core.abstract import AbstractDetector, AbstractOneStageDetector
from nndet.core.boxes.matcher import Matcher
from nndet.core.boxes.sampler import SamplerType
from nndet.core.post.box import BoxPostprocessing
from nndet.core.rcnn import RCNN
from nndet.core.rois.module import RoIModule
from nndet.core.rois.pooler import RoIPooler
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.mixins.evaluation import BoxWithRPNEvalMixin
from nndet.ptmodule.mixins.model import TwoStageMixin
from nndet.ptmodule.mixins.prediction import BoxPredictionMixin
from nndet.ptmodule.mixins.prepare import BoxesPrepareMixin
from nndet.ptmodule.module import LightningBaseModule


@MODULE_REGISTRY.register
class FasterRCNNModule(
    LightningBaseModule,  # Detection Base
    BoxesPrepareMixin,  # prepare batch for box training
    BoxWithRPNEvalMixin,  # Bounding Box Evaluation (with RPN)
    TwoStageMixin,  # Single Stage Detector
    BoxPredictionMixin,  # Bounding Box Sweep
):
    full_detector_cls: Type[AbstractDetector] = RCNN  # Two stage detector class RCNN
    # Use `detector_cls` to set RPN module class
    # define RPN cls
    detector_cls: Type[AbstractOneStageDetector] = ...

    ###################
    # RPN Configuration
    ###################
    backbone_cls: Type[AbstractEncoder] = ...  # define class for backbone
    backbone_conv_cls: Type[BaseConvNormAct] = ...  # conv class used for backbone
    backbone_block: Type[
        AbstractBlock
    ] = ...  # define central building block of backbone

    neck_cls: Type[BaseUFPN] = ...  # define class for neck
    neck_conv_cls: Type[BaseConvNormAct] = ...  # conv class used for neck

    head_cls: Type[AnchorHead] = ...  # define class for head
    head_conv_cls: Type[BaseConvNormAct] = ...  # conv class used for head
    head_classifier_cls: Type[DenseClassifier] = ...  # define class for head classifier
    head_regressor_cls: Type[DenseRegressor] = ...  # define class for head regressor
    # [optional] sampler class for negative mining
    # if None: no sampler will be given to the head
    head_sampler_cls: Type[SamplerType] = ...

    matcher_cls: Type[Matcher] = ...  # define class to match anchors to ground truth
    # Not supprted here; see `MaskRCNN`
    segmenter_cls: Optional[Type[Segmenter]] = None

    ########################
    # RoI Head Configuration
    ########################
    # RoI classes
    roi_conv_cls = ...  # conv class used for RoI head
    roi_module_cls: Type[RoIModule] = ...  # class of RoI module
    roi_head_cls: Type[RoIBoxHead] = ...  # class of box head of RoI module
    roi_classifier_cls: Type[RoIClassifier] = ...  # box head classifier class
    roi_regressor_cls: Type[RoIRegressor] = ...  # box head regressor class

    roi_matcher_cls: Type[Matcher] = ...  # class of RoI matcher
    roi_sampler_cls: Type[SamplerType] = ...  # class of RoI sampler
    roi_box_pooler_cls: Type[RoIPooler] = ...  # class of RoI box pooler
    roi_box_post_cls: Type[
        BoxPostprocessing
    ] = ...  # define roi box postprocessing strategy

    # Not supprted here; see `MaskRCNN`
    roi_masker_cls = None  # class of RoI mask head
    roi_mask_pooler_cls = None  # class of RoI mask pooler
    roi_mask_post_cls = None
