from typing import Mapping, Type

from nndet.ptmodule.module import LightningBaseModule
from nndet.utils.registry import Registry

MODULE_REGISTRY: Mapping[str, Type[LightningBaseModule]] = Registry()

from nndet.ptmodule.frcnn import FasterRCNNModule
from nndet.ptmodule.mrcnn import MaskRCNNModule

# register modules
from nndet.ptmodule.retinanet import RetinaNetC001
from nndet.ptmodule.retinaunet import RetinaUNetV001
