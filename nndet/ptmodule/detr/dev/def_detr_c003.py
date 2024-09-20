# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Type

from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.backbone.blueprints.resconv import ResConvWithPoolBackbone
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.detr.boxdetr_v002 import BoxDeformableDETRV002


@MODULE_REGISTRY.register
class BoxDeformableDETRC003(BoxDeformableDETRV002):
    pass


@MODULE_REGISTRY.register
class ResPBoxDeformableDETRC003(BoxDeformableDETRV002):
    backbone_cls: Type[AbstractBackbone] = ResConvWithPoolBackbone  # define class for backbone
