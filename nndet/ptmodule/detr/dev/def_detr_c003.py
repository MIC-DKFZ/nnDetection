# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.detr.boxdetr_v002 import BoxDeformableDETRV002


@MODULE_REGISTRY.register
class BoxDeformableDETRC003(BoxDeformableDETRV002):
    pass
