# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Mapping, Type

from nndet.io.augmentation.base import AugmentationSetup
from nndet.utils.registry import Registry

AUGMENTATION_REGISTRY: Mapping[str, Type[AugmentationSetup]] = Registry()

import nndet.io.augmentation.pipeline
