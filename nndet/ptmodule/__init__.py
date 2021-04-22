from typing import Mapping, Type
from nndet.utils.registry import Registry
from nndet.ptmodule.base_module import LightningBaseModule
MODULE_REGISTRY: Mapping[str, Type[LightningBaseModule]] = Registry()

from nndet.ptmodule.retinaunet import *
