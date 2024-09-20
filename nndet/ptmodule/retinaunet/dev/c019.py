# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.mixins.prediction.boxes import BoxPredictionMixinV3
from nndet.ptmodule.retinaunet.run_v002 import RetinaUNetFocalV002


@MODULE_REGISTRY.register
class RetinaUNetFocalC019(RetinaUNetFocalV002, BoxPredictionMixinV3):
    pass
