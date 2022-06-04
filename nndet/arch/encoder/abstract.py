# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Dict, List, TypeVar, Union

import torch
import torch.nn as nn

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


EncoderType = TypeVar("EncoderType", bound=AbstractEncoder)
