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
from typing import Callable, Tuple, Sequence, Union, List, Dict, Optional

from nndet.arch.encoder.abstract import AbstractEncoder
from nndet.arch.blocks.basic import AbstractBlock


__all__ = ["Encoder"]


class Encoder(AbstractEncoder):
    def __init__(self,
                 conv: Callable[[], nn.Module],
                 conv_kernels: Sequence[Union[Tuple[int], int]],
                 strides: Sequence[Union[Tuple[int], int]],
                 block_cls: AbstractBlock,
                 in_channels: int,
                 start_channels: int,
                 stage_kwargs: Sequence[dict] = None,
                 out_stages: Sequence[int] = None,
                 max_channels: int = None,
                 first_block_cls: Optional[AbstractBlock] = None,
                 ):
        """
        Build a modular encoder model with specified blocks
        The Encoder consists of "stages" which (in general) represent one
        resolution in the resolution pyramid. The first level alwasys has
        full resolution.

        Args:
            conv: conv generator to use for internal convolutions
            strides: strides for pooling layers. Should have one
                element less than conv_kernels
            conv_kernels: kernel sizes for convolutions
            block_cls: generate a block of convolutions (
                e.g. stacked residual blocks)
            in_channels: number of input channels
            start_channels: number of start channels
            stage_kwargs: additional keyword arguments for stages.
                Defaults to None.
            out_stages: define which stages should be returned. If `None` all
                stages will be returned.Defaults to None.
            first_block_cls: generate a block of convolutions for the first stage
                By default this equal the provided block_cls
        """
        super().__init__()
        self.num_stages = len(conv_kernels)
        self.dim = conv.dim
        if stage_kwargs is None:
            stage_kwargs = [{}] * self.num_stages
        elif isinstance(stage_kwargs, dict):
            stage_kwargs = [stage_kwargs] * self.num_stages
        assert len(stage_kwargs) == len(conv_kernels)

        if out_stages is None:
            self.out_stages = list(range(self.num_stages))
        else:
            self.out_stages = out_stages
        if first_block_cls is None:
            first_block_cls = block_cls

        stages = []
        self.out_channels = []
        if isinstance(strides[0], int):
            strides = [tuple([s] * self.dim) for s in strides]
        self.strides = strides
        for stage_id in range(self.num_stages):
            if stage_id == 0:
                _block = first_block_cls(
                    conv=conv,
                    in_channels=in_channels,
                    out_channels=start_channels,
                    conv_kernel=conv_kernels[stage_id],
                    stride=None,
                    max_out_channels=max_channels,
                    **stage_kwargs[stage_id],
                )
            else:
                _block = block_cls(
                    conv=conv,
                    in_channels=in_channels,
                    out_channels=None,
                    conv_kernel=conv_kernels[stage_id],
                    stride=strides[stage_id - 1],
                    max_out_channels=max_channels,
                    **stage_kwargs[stage_id],
                )
            in_channels = _block.get_output_channels()
            self.out_channels.append(in_channels)
            stages.append(_block)
        self.stages = torch.nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward data through encoder
        
        Args:
            x: input data
        
        Returns:
            List[torch.Tensor]: list of output from stages defined by
                param:`out_stages`
        """
        outputs = []
        for stage_id, module in enumerate(self.stages):
            x = module(x)
            if stage_id in self.out_stages:
                outputs.append(x)
        return outputs

    def get_channels(self) -> List[int]:
        """
        Compute number of channels for each returned feature map inside the forward pass

        Returns
            list: list with number of channels corresponding to returned feature maps
        """
        out_channels = []
        for stage_id in range(self.num_stages):
            if stage_id in self.out_stages:
                out_channels.append(self.out_channels[stage_id])
        return out_channels

    def get_strides(self) -> List[List[int]]:
        """
        Compute number backbone strides for 2d and 3d case and all options of network

        Returns
            List[List[int]]: defines the absolute stride for each output
                feature map with respect to input size
        """
        out_strides = []
        for stage_id in range(self.num_stages):
            if stage_id == 0:
                out_strides.append([1] * self.dim)
            else:
                new_stride = [prev_stride * pool_size for prev_stride, pool_size
                              in zip(out_strides[stage_id - 1], self.strides[stage_id - 1])]
                out_strides.append(new_stride)
        return out_strides
