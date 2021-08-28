from typing import Mapping, Type

from nndet.planning.experiment.base import AbstractPlanner, PlannerType
from nndet.utils.registry import Registry

PLANNER_REGISTRY: Mapping[str, Type[PlannerType]] = Registry()

from nndet.planning.experiment.dev import *
from nndet.planning.experiment.v001 import D3V001
