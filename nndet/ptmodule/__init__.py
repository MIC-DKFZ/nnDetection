from typing import Mapping, Type
from nndet.utils.registry import Registry
from nndet.ptmodule.base_module import LightningBaseModule
MODULE_REGISTRY: Mapping[str, Type[LightningBaseModule]] = Registry()

# register modules
from nndet.ptmodule.retinaunet import * 
