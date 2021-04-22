from typing import Mapping, Type
from nndet.io.augmentation.base import AugmentationSetup
from nndet.utils.registry import Registry
AUGMENTATION_REGISTRY: Mapping[str, Type[AugmentationSetup]] = Registry()

from nndet.io.augmentation.bg_aug import (
    NoAug,
    DefaultAug,
    BaseMoreAug,
    MoreAug,
    InsaneAug,
    )
