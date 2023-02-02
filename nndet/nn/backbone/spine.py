from typing import List, Optional

import torch

from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.neck.abstract import AbstractNeck
from nndet.utils.typing import ND_TUPLE_INT


class SpineWrapper(AbstractBackbone):
    def __init__(
        self,
        backbone: AbstractBackbone,
        neck: AbstractNeck,
    ) -> None:
        """
        Wrap a backbone and neck module to provide
        methods from the backbone module.

        Args:
            backbone: backbone for feature extraction
            neck: for feature recombination
        """
        super().__init__()
        self.backbone = backbone
        self.neck = neck

    def forward(
        self,
        images: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Forward input through network

        Args
            x: input tensor

        Returns
            list: list with feature maps from multiple resolutions
                Sorted from P0 (highest res) to PX (lowest res)
        """
        return self.neck(self.backbone(images))

    @classmethod
    def from_config_plan(
        cls,
        backbone_cfg: dict,
        plan_arch: dict,
    ):
        """
        Instantiate Backbone from given configs

        Not supported for Spine module

        Args
            backbone_cfg: backbone configuration
            plan_arch: arguments provided plan
        """
        raise RuntimeError("SpineWrapper is a special backbne and can not be instantiated from a config.")

    def get_channels(self) -> List[Optional[int]]:
        """
        Compute number of channels for each returned feature map
        inside the forward pass

        Returns
            List[int]: list with number of channels corresponding to
                returned feature maps. Undefined levels will be `None`.
        """
        return self.neck.get_channels()

    def get_relative_strides(self) -> List[Optional[ND_TUPLE_INT]]:
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
        return self.backbone.get_relative_strides()
