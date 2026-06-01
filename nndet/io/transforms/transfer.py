# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import torch

from nndet.io.transforms.base import AbstractTransform


class TransferInputChannel(AbstractTransform):
    def __init__(
        self,
        out_channels: int,
        data_key: str,
    ):
        """
        Create a new output tensor with more channels and save the data
        to a random channel

        Args:
            out_channels: number of output channels
            data_key: key where data is located

        Raises:
            ValueError: `out_channels` needs to be an integer
            ValueError: `data_key` needs to a be str
        """
        super().__init__(grad=False)
        if not isinstance(out_channels, int):
            raise ValueError("Exptected in_channels of type int received " f"{type(out_channels)} : {out_channels}")
        if not isinstance(data_key, str):
            raise ValueError("Exptected in_channels of type int received " f"{type(data_key)} : {data_key}")
        self.out_channels = out_channels
        self.data_key = data_key

    def forward(self, **data) -> dict:
        in_data = data[self.data_key]
        _in_shape = list(in_data.shape)
        _in_shape[1] = self.out_channels
        out_data = torch.zeros(
            tuple(_in_shape),
            dtype=in_data.dtype,
            device=in_data.device,
        )
        c = torch.randint(low=0, high=self.out_channels, size=(1,)).item()
        out_data[:, [c]] = in_data
        data[self.data_key] = out_data
        return data
