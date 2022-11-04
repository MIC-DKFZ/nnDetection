# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Mapping, Type

from nndet.ptmodule.module import LightningBaseModule
from nndet.utils.registry import Registry

MODULE_REGISTRY: Mapping[str, Type[LightningBaseModule]] = Registry()

# register modules

from nndet.ptmodule.detr import DETRModule
from nndet.ptmodule.frcnn import FasterRCNNModule
from nndet.ptmodule.mrcnn import MaskRCNNModule
from nndet.ptmodule.retinanet import RetinaNetModule
from nndet.ptmodule.retinaunet import RetinaUNetModule
