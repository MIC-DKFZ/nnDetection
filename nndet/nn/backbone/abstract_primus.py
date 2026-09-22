# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

import torch


class WrapperPrimusAbstractBackbone(torch.nn.Module):
    @abstractmethod
    def forward(self, x) -> List[torch.Tensor]:
        raise NotImplementedError

    @classmethod
    def from_config_plan(
        cls,
        backbone_cfg: Dict,
        plan_arch: Dict,
    ):
        """
        Instantiate Backbone from given configs.

        Args:
            backbone_cfg: backbone configuration
            plan_arch: arguments provided plan
        """
        raise NotImplementedError




class PrimusAbstract(torch.nn.Module):

    @abstractmethod
    def forward(self, x) -> List[torch.Tensor]:
        """
        Forward input through network

        Args
            x: input tensor

        Returns
            list: list with feature maps from multiple resolutions
                Sorted from P0 (highest res) to PX (lowest res)
        """
        raise NotImplementedError

    @abstractmethod
    def get_channels(self) -> List[Optional[int]]:
        raise NotImplementedError

    @abstractmethod
    def get_relative_strides(self) -> List[Optional[Tuple[int]]]:
        """
        Retrieve relative strides of the backbone feature maps.
        Starting with the highest resolution feature map to the lowest
        resolution feature map. Usually the first feature map will have stride
        1.

        Returns
            List[Tuple[int]]: defines the absolute stride for each output
                feature map with respect to input size. Undefined levels
                will be `None`.
        """
        raise NotImplementedError

    def get_absolute_strides(self) -> List[Optional[Tuple[int]]]:
        return None
    @abstractmethod
    def check_patch_size(self, patch_size: Sequence[int]) -> bool:
        return True



