# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Type

from nndet.core.boxes.matcher import ATSSMatcher, IoUMatcher
from nndet.core.boxes.sampler import (
    BalancedHardNegativeSampler,
    HardNegativeSamplerBatched,
)
from nndet.core.post.box import BoxPostprocessing, CrossLevelBoxPostprocessing
from nndet.core.post.mask import NoMaskPostprocessing
from nndet.core.rcnn import RCNN
from nndet.core.retina import BaseRetinaNet
from nndet.core.rois.module import RoIModule
from nndet.core.rois.pooler import RoIAlignNaiveAssign
from nndet.nn.backbone.blueprints.conv import ConvBackbone
from nndet.nn.heads.classifier.dense import BCECLassifier
from nndet.nn.heads.classifier.roi import (
    BCEConvRoIClassifier,
    BCEFCRoIClassifier,
    CEConvRoIClassifier,
)
from nndet.nn.heads.comb.anchor_sampled import BoxHeadHNM
from nndet.nn.heads.comb.roi import RoIBoxHead
from nndet.nn.heads.masker import BCESingleMasker
from nndet.nn.heads.masker.base import BDiceBCESingleMasker
from nndet.nn.heads.regressor.dense import L1Regressor
from nndet.nn.heads.regressor.roi import L1ConvRoIRegressor, L1FCRoIRegressor
from nndet.nn.heads.segmenter import DiCESegmenterFgBg
from nndet.nn.layers.conv import ConvGroupLReLU, ConvInstanceLReLU
from nndet.nn.neck.fpn import UFPN
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.mrcnn.m001 import MaskURCNNModule


@MODULE_REGISTRY.register
class MaskRCNNC001(MaskURCNNModule):
    # Use `detector_cls` to set RPN module class
    full_detector_cls = RCNN  # Two stage detector class RCNN
    # define detector cls
    detector_cls = BaseRetinaNet

    backbone_cls = ConvBackbone  # define class for backbone
    backbone_conv_cls = ConvInstanceLReLU  # conv class used for backbone

    neck_cls = UFPN  # define class for neck
    neck_conv_cls = ConvInstanceLReLU  # conv class used for neck

    head_cls = BoxHeadHNM  # define class for head
    head_conv_cls = ConvGroupLReLU  # conv class used for head
    head_classifier_cls = BCECLassifier  # define class for head classifier
    head_regressor_cls = L1Regressor  # define class for head regressor
    # [optional] sampler class for negative mining
    # if None: no sampler will be given to the head
    head_sampler_cls = HardNegativeSamplerBatched

    matcher_cls = ATSSMatcher  # define class to match anchors to ground truth
    box_post_cls: Type[BoxPostprocessing] = CrossLevelBoxPostprocessing  # define box postprocessing strategy
    segmenter_cls = None  # [optional] segmentation head as in RetinaUNet

    # RoI classes
    roi_conv_cls = ConvGroupLReLU
    roi_module_cls = RoIModule  # RoIModule
    roi_head_cls = RoIBoxHead  # RoIBoxHead
    roi_classifier_cls = BCEConvRoIClassifier  # RoIClassifierTwoMLP
    roi_regressor_cls = L1ConvRoIRegressor  # RoIRegressorConv

    roi_matcher_cls = IoUMatcher  # IoUMatcher
    roi_sampler_cls = BalancedHardNegativeSampler  # BalancedHardNegativeSampler
    roi_box_pooler_cls = RoIAlignNaiveAssign  # RoIAlignNaiveAssign
    roi_box_post_cls = CrossLevelBoxPostprocessing  #: define roi box postprocessing strategy

    # optional mask branches
    roi_masker_cls = BCESingleMasker  # BCESingleMasker
    roi_mask_pooler_cls = RoIAlignNaiveAssign  # RoIAlignNaiveAssign
    roi_mask_post_cls = NoMaskPostprocessing


@MODULE_REGISTRY.register
class MaskURCNNC001(MaskRCNNC001):
    segmenter_cls = DiCESegmenterFgBg  # [optional] segmentation head as in RetinaUNet


@MODULE_REGISTRY.register
class MaskURCNNC001RSB(MaskURCNNC001):
    segmenter_cls = DiCESegmenterFgBg  # [optional] segmentation head as in RetinaUNet
    roi_sampler_cls = HardNegativeSamplerBatched  # [optional] segmentation head as in RetinaUNet


@MODULE_REGISTRY.register
class MaskURCNNC001RSBCE(MaskURCNNC001):  # wrong parent class
    roi_classifier_cls = CEConvRoIClassifier


@MODULE_REGISTRY.register
class MaskURCNNC001RSBCEF(MaskURCNNC001RSB):  # fixed parent class
    segmenter_cls = DiCESegmenterFgBg  # [optional] segmentation head as in RetinaUNet
    roi_sampler_cls = HardNegativeSamplerBatched  # [optional] segmentation head as in RetinaUNet
    roi_classifier_cls = CEConvRoIClassifier


@MODULE_REGISTRY.register
class MaskURCNNC001FC(MaskURCNNC001):
    segmenter_cls = DiCESegmenterFgBg  # [optional] segmentation head as in RetinaUNet

    roi_classifier_cls = BCEFCRoIClassifier  # RoIClassifierTwoMLP
    roi_regressor_cls = L1FCRoIRegressor  # RoIRegressorConv


@MODULE_REGISTRY.register
class MaskURCNNC001FCRSB(MaskURCNNC001FC):
    segmenter_cls = DiCESegmenterFgBg  # [optional] segmentation head as in RetinaUNet

    roi_classifier_cls = BCEFCRoIClassifier  # RoIClassifierTwoMLP
    roi_regressor_cls = L1FCRoIRegressor  # RoIRegressorConv

    roi_sampler_cls = HardNegativeSamplerBatched  # [optional] segmentation head as in RetinaUNet


@MODULE_REGISTRY.register
class MaskURCNNC001DiceBCE(MaskURCNNC001):
    segmenter_cls = DiCESegmenterFgBg  # [optional] segmentation head as in RetinaUNet

    roi_masker_cls = BDiceBCESingleMasker


@MODULE_REGISTRY.register
class MaskURCNNC001RSBDiceBCE(MaskURCNNC001RSB):
    segmenter_cls = DiCESegmenterFgBg  # [optional] segmentation head as in RetinaUNet
    roi_sampler_cls = HardNegativeSamplerBatched  # [optional] segmentation head as in RetinaUNet

    roi_masker_cls = BDiceBCESingleMasker
