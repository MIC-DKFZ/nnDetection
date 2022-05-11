from abc import abstractclassmethod, abstractmethod
from typing import List

import torch

from nndet.utils.typing import ND_TUPLE_INT


class AbstractBackbone(torch.nn.Module):
    @abstractmethod
    def forward(self, x) -> List[torch.Tensor]:
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

    @abstractmethod
    def get_channels(self) -> List[int]:
        """
        Compute number of channels for each returned feature map
        inside the forward pass

        Returns
            List[int]: list with number of channels corresponding to
                returned feature maps
        """
        raise NotImplementedError

    @abstractmethod
    def get_strides(self) -> List[ND_TUPLE_INT]:
        """
        Retrieve absolute strides of the backbone feature maps
        (ordered from lowest to highest strides)

        Returns
            List[List[int]]: defines the absolute stride for each output
                feature map with respect to input size
        """
        raise NotImplementedError
