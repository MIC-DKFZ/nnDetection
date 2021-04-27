"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Union, TypeVar
from abc import abstractmethod


__all__ = ["AbstractEncoder"]


class AbstractEncoder(nn.Module):
    def __int__(self, **kwargs):
        """
        Provides an abstract interface for backbone networks
        """
        super().__init__(**kwargs)

    @abstractmethod
    def forward(self, x) -> List[torch.Tensor]:
        """
        Forward input through network

        Args
            x (torch.tensor): input tensor

        Returns
            list: list with feature maps from multiple resolutions
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
    def get_strides(self) -> List[Dict[str, Union[List[int], int]]]:
        """
        Compute number backbone strides for 2d and 3d case and all options 
        of network

        Returns
            List[Dict[str, Union[List[int], int]]]: dict with 'xy' for 2d 
                stride and optional 'z' for 3d cases. List
                describes stride at respective output level
        """
        raise NotImplementedError


EncoderType = TypeVar('EncoderType', bound=AbstractEncoder)
