from typing import Mapping, Type

from nndet.ptmodule.module import LightningBaseModule
from nndet.utils.registry import Registry

MODULE_REGISTRY: Mapping[str, Type[LightningBaseModule]] = Registry()

from nndet.ptmodule.retinanet import *

# register modules
from nndet.ptmodule.retinaunet import *
