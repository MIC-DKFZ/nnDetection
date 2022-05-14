from abc import abstractclassmethod, abstractmethod
from typing import List, Sequence

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
        (ordered from lowest to highest strides -> i.e. highest to
        lowest resolution)

        Returns
            List[List[int]]: defines the absolute stride for each output
                feature map with respect to input size
        """
        raise NotImplementedError

    def check_patch_size(self, patch_size: Sequence[int]) -> bool:
        """
        Check if the provided patch_size works with this network config

        Args:
            patch_size: patch size to check

        Raises:
            ValueError: raised only if network dimensions and patch size
                dimensions are not consistent (e.g. 2D net with 3D patch)

        Returns:
            bool: `True` if patch size is compatible with network config,
                `False` otherwise.
        """
        absolute_strides = self.get_strides()
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
