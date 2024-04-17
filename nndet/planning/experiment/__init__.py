# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Mapping, Type

from nndet.planning.experiment.base import AbstractPlanner
from nndet.utils.registry import Registry

PLANNER_REGISTRY: Mapping[str, Type[AbstractPlanner]] = Registry()

from nndet.planning.experiment.dev import D2C004, D3V001AEP, D3V001FP16I16
from nndet.planning.experiment.v001 import D3V001
from nndet.planning.experiment.v002 import D3C010  # D3V002
