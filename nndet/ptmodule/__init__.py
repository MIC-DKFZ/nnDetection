# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Mapping, Type

from nndet.ptmodule.module import LightningBaseModule
from nndet.utils.registry import Registry

MODULE_REGISTRY: Mapping[str, Type[LightningBaseModule]] = Registry()

# register modules

from nndet.ptmodule.detr import BoxDETRModule
from nndet.ptmodule.detr.ResEnc_BoxDETR_v002 import (
    BoxDeformableDETRV002_ResEnc,
    BoxDeformableDETRV002_ResEnc_dyn,
    BoxDeformableDETRV002_ResEnc_TL,
    BoxDeformableDETRV002_ResEnc_dyn_TL,
    BoxDeformableDETRV002_ResEnc_TL_warmuptransformer_head,
    DetSeg_DeformableDETR_ResEnc,
)
from nndet.ptmodule.detr.Primus_BoxDETR_v0002 import BoxDeformableDETRV002_Primus, BoxDeformableDETRV002_Primus_TL
from nndet.ptmodule.retinanet import RetinaNetModule
from nndet.ptmodule.retinanet2s import RetinaNet2SModule
from nndet.ptmodule.retinaunet import RetinaUNetModule
from nndet.ptmodule.retinaunet.ResEncModels import *
from nndet.ptmodule.retinaunet2sm import RetinaUNet2SMModule
