# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Optional, Sequence


class ModelMixin(ABC):
    @classmethod
    @abstractmethod
    def from_config_plan(
        cls,
        model_cfg: dict,
        plan_arch: dict,
        plan_anchors: dict,
        patch_size: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        """
        Create Configurable Model

        Args:
            model_cfg: model configurations.
                Exact parameters depend on subclass.
            plan_arch: plan architecture
                Exact parameters depend on subclass.
            plan_anchors: parameters for anchors
                Exact parameters depend on subclass.
            patch_size: optionally provide the patch size
                to check compatibility with backbone
            **kwargs: ignored
        """
        raise NotImplementedError
