# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod
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
