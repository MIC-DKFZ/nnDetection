# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

import torch


class WrapperResEncAbstractBackbone(torch.nn.Module):
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


class ResEncAbstract(torch.nn.Module):

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
        """
        Retrieve absolute strides of the backbone feature maps
        (ordered from lowest to highest strides -> highest to
        lowest resolution)

        Returns
            List[Tuple[int]]: defines the absolute stride for each output
                feature map with respect to input size. Undefined levels
                will be `None`.
        """
        relative_strides = self.get_relative_strides()
        absolute_strides = []
        new_stride = None
        for level_idx in range(len(relative_strides)):
            if relative_strides[level_idx] is None:
                absolute_strides.append(None)
                continue

            if new_stride is None:
                new_stride = relative_strides[level_idx]
            else:
                new_stride = [ns * s for ns, s in zip(new_stride, relative_strides[level_idx])]
            absolute_strides.append(tuple(new_stride))
        assert len(relative_strides) == len(absolute_strides)
        return absolute_strides

    def check_patch_size(self, patch_size: Sequence[int]) -> bool:
        """
        Check if the provided patch_size works with this network config

        Args:
            patch_size: patch size to check (without channels and batch dim)

        Raises:
            ValueError: raised only if network dimensions and patch size
                dimensions are not consistent (e.g. 2D net with 3D patch)

        Returns:
            bool: `True` if patch size is compatible with network config,
                `False` otherwise.
        """
        absolute_strides = self.get_absolute_strides()
        max_stride = absolute_strides[-1]
        patch_dim = len(patch_size)
        if isinstance(max_stride, int):
            per_axis = [(ps % max_stride) == 0 for ps in patch_size]
        else:
            stride_dim = len(max_stride)
            if patch_dim != stride_dim:
                raise ValueError(
                    "Found inconsistent patch and stride dimensions: "
                    f"max stride {max_stride} and patch size {patch_size}"
                )
            per_axis = [(ps % ms) == 0 for ps, ms in zip(patch_size, max_stride)]
        return all(per_axis)
