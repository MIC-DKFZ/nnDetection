from typing import Iterable, Mapping
from nndet.utils.registry import Registry

DATALOADER_REGISTRY: Mapping[str, Iterable] = Registry()

from nndet.io.datamodule.bg_loader import (
    DataLoader3DFast,
    DataLoader3DBalanced,
    DataLoader3DOffset,
    DataLoader2DOffset,
    DataLoader2DFast,
    DataLoader2DDeeplesion,
)
