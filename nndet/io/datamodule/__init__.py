# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Iterable, Mapping

from nndet.utils.registry import Registry

DATALOADER_REGISTRY: Mapping[str, Iterable] = Registry()

from nndet.io.datamodule.loader import BaseDataLoader2D, DataLoader3D
