from typing import Iterable, Mapping

from nndet.utils.registry import Registry

DATALOADER_REGISTRY: Mapping[str, Iterable] = Registry()

from nndet.io.datamodule.loader import BaseDataLoader2D, DataLoader3D
