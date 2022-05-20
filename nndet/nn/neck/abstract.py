from __future__ import annotations

from abc import abstractclassmethod, abstractmethod
from typing import List

import torch


class AbstractNeck(torch.nn.Module):
    @abstractmethod
    def forward(
        self,
        backbone_output: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Forward input through network

        Args
            x: input tensor

        Returns
            list: list with feature maps from multiple resolutions
                Sorted from P0 (highest res) to PX (lowest res)
        """
        raise NotImplementedError

    @abstractclassmethod
    def from_config_plan(
        cls,
        backbone_cfg: dict,
        plan_arch: dict,
    ) -> AbstractNeck:
        """
        Instantiate Backbone from given configs

        Args
            backbone_cfg: backbone configuration
            plan_arch: arguments provided plan
        """
        raise NotImplementedError
