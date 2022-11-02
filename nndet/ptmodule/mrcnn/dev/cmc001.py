# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Type

from nndet.core.boxes.matcher import ATSSMatcher, IoUMatcher
from nndet.core.boxes.sampler import (
    BalancedHardNegativeSampler,
    HardNegativeSamplerBatched,
)
from nndet.core.post.box import BoxPostprocessing, CrossLevelBoxPostprocessing
from nndet.core.post.mask import MaskPostprocessing, NoMaskPostprocessing
from nndet.core.rcnn import RCNN
from nndet.core.retina import BaseRetinaNet
from nndet.core.rois.module import CascadeRoIModule
from nndet.core.rois.pooler import RoIAlignNaiveAssign
from nndet.nn.backbone.blueprints.conv import ConvBackbone
from nndet.nn.heads.classifier.dense import BCECLassifier
from nndet.nn.heads.classifier.roi import CEConvRoIClassifier
from nndet.nn.heads.comb.anchor_sampled import BoxHeadHNM
from nndet.nn.heads.comb.roi import RoIBoxHead
from nndet.nn.heads.masker import BCESingleMasker
from nndet.nn.heads.regressor.dense import L1Regressor
from nndet.nn.heads.regressor.roi import L1ConvRoIRegressor
from nndet.nn.heads.segmenter import DiCESegmenterFgBg
from nndet.nn.layers.conv import ConvGroupLReLU, ConvInstanceLReLU
from nndet.nn.neck.fpn import UFPN
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.mrcnn.cm001 import CascadeMaskURCNNModule


@MODULE_REGISTRY.register
class CascadeMaskURCNNC001(CascadeMaskURCNNModule):
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
    box_post_cls: Type[
        BoxPostprocessing
    ] = CrossLevelBoxPostprocessing  # define box postprocessing strategy
    segmenter_cls = DiCESegmenterFgBg  # [optional] segmentation head as in RetinaUNet

    # RoI classes
    roi_conv_cls = ConvGroupLReLU
    roi_module_cls = CascadeRoIModule  # RoIModule
    roi_head_cls = RoIBoxHead  # RoIBoxHead
    roi_classifier_cls = CEConvRoIClassifier  # RoIClassifierTwoMLP
    roi_regressor_cls = L1ConvRoIRegressor  # RoIRegressorConv
    roi_box_post_cls = CrossLevelBoxPostprocessing

    roi_matcher_cls = IoUMatcher  # IoUMatcher
    roi_sampler_cls = BalancedHardNegativeSampler  # BalancedHardNegativeSampler
    roi_box_pooler_cls = RoIAlignNaiveAssign  # RoIAlignNaiveAssign

    # optional mask branches
    roi_masker_cls = BCESingleMasker  # BCESingleMasker
    roi_mask_pooler_cls = RoIAlignNaiveAssign  # RoIAlignNaiveAssign
    roi_mask_post_cls: Type[
        MaskPostprocessing
    ] = NoMaskPostprocessing  # define roi mask postprocessing strategy
