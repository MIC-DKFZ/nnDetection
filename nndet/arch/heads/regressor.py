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

from typing import Optional, Tuple, Callable, TypeVar
from abc import abstractmethod

from loguru import logger

from nndet.core.boxes import box_iou
from nndet.arch.layers.scale import Scale
from torch import Tensor

from nndet.losses import SmoothL1Loss, GIoULoss


CONV_TYPES = (nn.Conv2d, nn.Conv3d)


class Regressor(nn.Module):
    @abstractmethod
    def compute_loss(self, pred_deltas: Tensor, target_deltas: Tensor, **kwargs) -> Tensor:
        """
        Compute regression loss (l1 loss)

        Args:
            pred_deltas (Tensor): predicted bounding box deltas [N,  dim * 2]
            target_deltas (Tensor): target bounding box deltas [N,  dim * 2]

        Returns:
            Tensor: loss
        """
        raise NotImplementedError


class BaseRegressor(Regressor):
    def __init__(self,
                 conv,
                 in_channels: int,
                 internal_channels: int,
                 anchors_per_pos: int,
                 num_levels: int,
                 num_convs: int = 3,
                 add_norm: bool = True,
                 learn_scale: bool = False,
                 **kwargs,
                 ):
        """
        Base class to build regressor heads with typical conv structure
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            anchors_per_pos: number of anchors per position
            num_levels: number of decoder levels which are passed through the
                regressor
            num_convs: number of convolutions
                in conv -> num convs -> final conv
            add_norm: en-/disable normalization layers in internal layers
            learn_scale: learn additional single scalar values per feature
                pyramid level
            kwargs: keyword arguments passed to first and internal convolutions
        """
        super().__init__()
        self.dim = conv.dim
        self.num_levels = num_levels
        self.num_convs = num_convs
        self.learn_scale = learn_scale

        self.anchors_per_pos = anchors_per_pos

        self.in_channels = in_channels
        self.internal_channels = internal_channels

        self.conv_internal = self.build_conv_internal(conv, add_norm=add_norm, **kwargs)
        self.conv_out = self.build_conv_out(conv)

        if self.learn_scale:
            self.scales = self.build_scales()

        self.loss: Optional[nn.Module] = None
        self.init_weights()

    def build_conv_internal(self, conv, **kwargs):
        """
        Build internal convolutions
        """
        _conv_internal = nn.Sequential()
        _conv_internal.add_module(
            name="c_in",
            module=conv(
                self.in_channels,
                self.internal_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                **kwargs,
            ))
        for i in range(self.num_convs):
            _conv_internal.add_module(
                name=f"c_internal{i}",
                module=conv(
                    self.internal_channels,
                    self.internal_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    **kwargs,
                ))
        return _conv_internal

    def build_conv_out(self, conv):
        """
        Build final convolutions
        """
        out_channels = self.anchors_per_pos * self.dim * 2
        return conv(
            self.internal_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            add_norm=False,
            add_act=False,
            bias=True,
        )

    def build_scales(self) -> nn.ModuleList:
        """
        Build additionales scalar values per level
        """
        logger.info("Learning level specific scalar in regressor")
        return nn.ModuleList([Scale() for _ in range(self.num_levels)])

    def forward(self, x: torch.Tensor, level: int, **kwargs) -> torch.Tensor:
        """
        Forward input

        Args:
            x: input feature map of size [N x C x Y x X x Z]

        Returns:
            torch.Tensor: classification logits for each anchor
                [N, n_anchors, dim*2]
        """
        bb_logits = self.conv_out(self.conv_internal(x))

        if self.learn_scale:
            bb_logits = self.scales[level](bb_logits)

        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        bb_logits = bb_logits.permute(*axes)
        bb_logits = bb_logits.contiguous()
        bb_logits = bb_logits.view(x.size()[0], -1, self.dim * 2)
        return bb_logits

    def compute_loss(self,
                     pred_deltas: Tensor,
                     target_deltas: Tensor,
                     **kwargs,
                     ) -> Tensor:
        """
        Compute regression loss (l1 loss)

        Args:
            pred_deltas: predicted bounding box deltas [N,  dim * 2]
            target_deltas: target bounding box deltas [N,  dim * 2]

        Returns:
            Tensor: loss
        """
        return self.loss(pred_deltas, target_deltas, **kwargs)

    def init_weights(self) -> None:
        """
        Init weights with normal distribution (mean=0, std=0.01)
        """
        logger.info("Overwriting regressor conv weight init")
        for layer in self.modules():
            if isinstance(layer, CONV_TYPES):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)


class L1Regressor(BaseRegressor):
    def __init__(self,
                 conv,
                 in_channels: int,
                 internal_channels: int,
                 anchors_per_pos: int,
                 num_levels: int,
                 num_convs: int = 3,
                 add_norm: bool = True,
                 beta: float = 1.,
                 reduction: Optional[str] = "sum",
                 loss_weight: float = 1.,
                 learn_scale: bool = False,
                 **kwargs,
                 ):
        """
        Build regressor heads with typical conv structure and smooth L1 loss
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            anchors_per_pos: number of anchors per position
            num_levels: number of decoder levels which are passed through the
                regressor
            num_convs: number of convolutions
                in conv -> num convs -> final conv
            add_norm: en-/disable normalization layers in internal layers
            beta: L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            loss_weight: scalar to balance multiple losses
            learn_scale: learn additional single scalar values per feature
                pyramid level
            kwargs: keyword arguments passed to first and internal convolutions
        """
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            internal_channels=internal_channels,
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            num_convs=num_convs,
            add_norm=add_norm,
            learn_scale=learn_scale,
            **kwargs
        )
        self.loss = SmoothL1Loss(
            beta=beta,
            reduction=reduction,
            loss_weight=loss_weight,
            )


class GIoURegressor(BaseRegressor):
    def __init__(self,
                 conv,
                 in_channels: int,
                 internal_channels: int,
                 anchors_per_pos: int,
                 num_levels: int,
                 num_convs: int = 3,
                 add_norm: bool = True,
                 reduction: Optional[str] = "sum",
                 loss_weight: float = 1.,
                 learn_scale: bool = False,
                 **kwargs,
                 ):
        """
        Build regressor heads with typical conv structure and generalized
        IoU loss
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            anchors_per_pos: number of anchors per position
            num_levels: number of decoder levels which are passed through the
                regressor
            num_convs: number of convolutions
                in conv -> num convs -> final conv
            add_norm: en-/disable normalization layers in internal layers
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            loss_weight: scalar to balance multiple losses
            learn_scale: learn additional single scalar values per feature
                pyramid level
            kwargs: keyword arguments passed to first and internal convolutions
        """
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            internal_channels=internal_channels,
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            num_convs=num_convs,
            add_norm=add_norm,
            learn_scale=learn_scale,
            **kwargs
        )
        self.loss = GIoULoss(
            reduction=reduction,
            loss_weight=loss_weight,
            )


RegressorType = TypeVar('RegressorType', bound=Regressor)
