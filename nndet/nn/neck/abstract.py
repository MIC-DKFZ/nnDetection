from abc import abstractclassmethod, abstractmethod
from typing import Dict

import torch


class AbstractNeck(torch.nn.Module):
    @abstractmethod
    def forward(self, inp: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward input through network

        Args
            x: input tensor

        Returns
            list: list with feature maps from multiple resolutions
        """
        raise NotImplementedError

    @abstractclassmethod
    def from_config_plan(
        cls,
        backbone_cfg: dict,
        plan_arch: dict,
    ):
        """
        Instantiate Backbone from given configs

        Args
            backbone_cfg: backbone configuration
            plan_arch: arguments provided plan
        """
        raise NotImplementedError
