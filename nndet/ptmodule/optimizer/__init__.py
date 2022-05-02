from typing import Any

from nndet.utils.registry import Registry

OPTIMIZER_REGISTRY: Registry[str, Any] = Registry()

from nndet.ptmodule.optimizer.adam import AdamWLWPoly
from nndet.ptmodule.optimizer.madgrad import MadgradLWPoly
from nndet.ptmodule.optimizer.radam import RAdamLWPoly
from nndet.ptmodule.optimizer.ranger import Ranger21, RangerLWPoly
from nndet.ptmodule.optimizer.sgd import SGDLWPoly
