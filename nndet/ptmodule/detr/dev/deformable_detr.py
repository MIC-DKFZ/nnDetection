# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Type

from nndet.core.boxes.criterions.base import ClassCriterion
from nndet.core.boxes.criterions.cls import FocalClassCriterionSigmoid
from nndet.core.post.detr import DETRBoxPost, TopKBoxPost
from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.backbone.blueprints.conv import ConvBackbone
from nndet.nn.heads.classifier.ffn import FFNClassifier, FocalFFNClassifier
from nndet.nn.heads.detr.base import DETRHead
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.detr.dev.c001 import BoxDETRC001


@MODULE_REGISTRY.register
class BoxDeformableDETRC001Focal(BoxDETRC001):
    transformer_cls = ...  # DeformableDETRTransformer
    backbone_cls: Type[AbstractBackbone] = ConvBackbone  #: define class for backbone

    head_cls: DETRHead = ...  # DeformableDETRHead  #: main DETR head
    head_classifier_cls: FFNClassifier = FocalFFNClassifier  #: define classifier class
    head_box_post_cls: DETRBoxPost = TopKBoxPost  #: define postprocessing strategy during inference
    matcher_class_criterion_cls: ClassCriterion = FocalClassCriterionSigmoid  #: criterion to compute class cost matrix
