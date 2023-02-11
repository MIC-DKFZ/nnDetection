# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import functools
from abc import abstractmethod
from typing import Optional, Sequence

import torch
from torch import Tensor

from nndet.losses.regression.giou import GIoULoss
from nndet.losses.regression.smoothl1 import SmoothL1Loss
from nndet.nn.heads.abstract import Regressor, RoIConv1x1View
from nndet.nn.layers.wrapper import nd_pool
from nndet.utils.typing import CONVGEN


class RoIRegressor(Regressor):
    def __init__(
        self,
        conv: CONVGEN,
        input_size: Sequence[int],
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        num_convs: int = 1,
        add_norm: bool = True,
        **kwargs,
    ):
        """
        Base class to build regressor heads with typical conv structure
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        Args:
            conv: Convolution modules which handles a single layer
            input_size: specify spatial dimensions of RoI
            in_channels: number of input channels
            internal_channels: number of channels internally used
            num_classes: number of foreground classes
            num_convs: number of convolutions
                input_conv -> num_convs -> output_convs
            add_norm: en-/disable normalization layers in internal layers
            kwargs: keyword arguments passed to first and internal convolutions

        Notes:
            `self.loss` needs to be overwritten in subclasses
        """
        super().__init__()
        self.dim = conv.dim
        self.num_convs = num_convs

        self.in_channels = in_channels
        self.internal_channels = internal_channels
        self.num_classes = num_classes
        self.input_size = input_size

        self.module_internal = self._build_module_internal(conv=conv, add_norm=add_norm, **kwargs)
        self.module_out = self._build_module_out(conv=conv)

        self.loss: Optional[torch.nn.Module] = None
        self.init_weights()

    @classmethod
    def class_agnostic(cls):
        """
        Indicate if RoI regressor produces per class regression deltas or not

        Returns:
            bool: `True` if regression deltas apply to all classes, `False`
                if per class regression deltas are computed
        """
        return True

    @abstractmethod
    def _build_module_internal(self, conv: CONVGEN, add_norm: bool, **kwargs) -> torch.nn.Module:
        """
        Build internal modules
        """
        raise NotImplementedError

    def _build_module_out(self, conv: CONVGEN) -> torch.nn.Module:
        """
        Build final convolution
        """
        if self.class_agnostic():
            _out = self.dim * 2
        else:
            _out = self.num_classes * self.dim * 2
        return conv(
            self.internal_channels,
            _out,
            kernel_size=1,
            stride=1,
            padding=0,
            add_norm=False,
            add_act=False,
            bias=True,
        )

    def init_weights(self):
        """
        Init weights with prior prob
        """
        pass

    def forward(self, features: Tensor) -> Tensor:
        """
        Forward input through module

        Args:
            features: input features [N, C, roi_size], where N=number of
                RoIs, C=number of channels, roi_size=spatial size of RoI

        Returns:
            torch.Tensor: predicted logits [N, dim * 2], where N=number of
                RoIs, dim=number of spatial dimensions
        """
        x = self.module_out(self.module_internal(features))
        return x.view(x.shape[0], -1)  # [N, C, 1] -> [N, C]

    def compute_loss(
        self,
        pred_deltas: Tensor,
        target_deltas: Tensor,
        target_labels: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Compute regression loss

        Args:
            pred_deltas: predicted bounding box deltas [N,  (num_classes *) dim * 2] where
                N=number of RoIs, dim=number of spatial dimeneions
            target_deltas: target bounding box deltas [N,  dim * 2] where
                N=number of RoIs, dim=number of spatial dimeneions
            target_labels: target labels for boxes [N], where
                N=number of RoIs
            kwargs: keyword arguments passed to loss function

        Returns:
            Tensor: loss
        """
        if not self.class_agnostic():
            # only compute loss on target class
            num_rois, _ = pred_deltas.shape
            _pred_deltas = pred_deltas.reshape(num_rois, self.num_classes, self.dim * 2)
            _target_labels = target_labels - 1  # matching adds +1 for background which needs to be removed
            _pred_deltas = _pred_deltas[torch.arange(num_rois), _target_labels]
        else:
            _pred_deltas = pred_deltas
        return self.loss(_pred_deltas, target_deltas, **kwargs)


class ConvRoIRegressor(RoIRegressor):
    def _build_module_internal(self, conv: CONVGEN, **kwargs) -> torch.nn.Module:
        """
        Build internal modules conv(s) -> pool -> out
        """
        _conv_internal = torch.nn.Sequential()
        _conv_internal.add_module(
            name="c_in",
            module=conv(
                self.in_channels,
                self.internal_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                **kwargs,
            ),
        )
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
                ),
            )
        _conv_internal.add_module(
            name="pool",
            module=nd_pool("AdaptiveAvg", self.dim, 1),
        )
        return _conv_internal


class FCRoIRegressor(RoIRegressor):
    def _build_module_internal(self, conv: CONVGEN, **kwargs) -> torch.nn.Module:
        """
        Build internal modules flatten -> FC(s) -> out
        """
        _conv_internal = torch.nn.Sequential()
        _conv_internal.add_module(
            name="conv1x1_view",
            module=RoIConv1x1View(dim=self.dim),
        )
        _conv_internal.add_module(
            name="c_in",
            module=conv(
                self.in_channels * functools.reduce(lambda a, b: a * b, self.input_size),
                self.internal_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                **kwargs,
            ),
        )
        for i in range(self.num_convs):
            _conv_internal.add_module(
                name=f"c_internal{i}",
                module=conv(
                    self.internal_channels,
                    self.internal_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    **kwargs,
                ),
            )
        return _conv_internal


class L1ConvRoIAgnosticRegressor(ConvRoIRegressor):
    def __init__(
        self,
        conv: CONVGEN,
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        input_size: Sequence[int],
        num_convs: int = 1,
        add_norm: bool = True,
        beta: float = 1.0,
        reduction: Optional[str] = "sum",
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ):
        """
        Regressor head with L1 loss. Structure: conv(s) -> pool -> out

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            num_classes: number of foreground classes
            input_size: specify spatial dimensions of RoI
            num_convs: number of convolutions
                input_conv -> num_convs -> output_convs
            add_norm: en-/disable normalization layers in internal layers
            beta: L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            kwargs: keyword arguments passed to first and internal convolutions
        """
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            internal_channels=internal_channels,
            input_size=input_size,
            num_convs=num_convs,
            add_norm=add_norm,
            num_classes=num_classes,
            **kwargs,
        )
        self.loss = SmoothL1Loss(
            beta=beta,
            reduction=reduction,
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
        )


class GIoUConvRoIAgnosticRegressor(ConvRoIRegressor):
    def __init__(
        self,
        conv: CONVGEN,
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        input_size: Sequence[int],
        num_convs: int = 1,
        add_norm: bool = True,
        reduction: Optional[str] = "sum",
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ):
        """
        Regressor head with GIoU loss. Structure: conv(s) -> pool -> out

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            num_classes: number of foreground classes
            input_size: specify spatial dimensions of RoI
            num_convs: number of convolutions
                input_conv -> num_convs -> output_convs
            add_norm: en-/disable normalization layers in internal layers
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            kwargs: keyword arguments passed to first and internal convolutions
        """
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            internal_channels=internal_channels,
            num_classes=num_classes,
            input_size=input_size,
            num_convs=num_convs,
            add_norm=add_norm,
            **kwargs,
        )
        self.loss = GIoULoss(
            reduction=reduction,
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
        )


class L1FCRoIAgnosticRegressor(FCRoIRegressor):
    def __init__(
        self,
        conv: CONVGEN,
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        input_size: Sequence[int],
        num_convs: int = 1,
        add_norm: bool = True,
        beta: float = 1.0,
        reduction: Optional[str] = "sum",
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ):
        """
        Regressor head with L1 loss. Structure: flatten -> FC(s) -> out

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            num_classes: number of foreground classes
            input_size: specify spatial dimensions of RoI
            num_convs: number of convolutions
                input_conv -> num_convs -> output_convs
            add_norm: en-/disable normalization layers in internal layers
            beta: L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            kwargs: keyword arguments passed to first and internal convolutions
        """
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            internal_channels=internal_channels,
            num_classes=num_classes,
            input_size=input_size,
            num_convs=num_convs,
            add_norm=add_norm,
            **kwargs,
        )
        self.loss = SmoothL1Loss(
            beta=beta,
            reduction=reduction,
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
        )


class GIoUFCRoIAgnosticRegressor(FCRoIRegressor):
    def __init__(
        self,
        conv: CONVGEN,
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        input_size: Sequence[int],
        num_convs: int = 1,
        add_norm: bool = True,
        reduction: Optional[str] = "sum",
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ):
        """
        Regressor head with GIoU loss. Structure: flatten -> FC(s) -> out

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            num_classes: number of foreground classes
            input_size: specify spatial dimensions of RoI
            num_convs: number of convolutions
                input_conv -> num_convs -> output_convs
            add_norm: en-/disable normalization layers in internal layers
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            kwargs: keyword arguments passed to first and internal convolutions
        """
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            internal_channels=internal_channels,
            num_classes=num_classes,
            input_size=input_size,
            num_convs=num_convs,
            add_norm=add_norm,
            **kwargs,
        )
        self.loss = GIoULoss(
            reduction=reduction,
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
        )


class L1ConvRoISpecificRegressor(L1ConvRoIAgnosticRegressor):
    @classmethod
    def class_agnostic(cls):
        """
        Indicate if RoI regressor produces per class regression deltas or not

        Returns:
            bool: `True` if regression deltas apply to all classes, `False`
                if per class regression deltas are computed
        """
        return False


class L1FCRoISpecificRegressor(L1FCRoIAgnosticRegressor):
    @classmethod
    def class_agnostic(cls):
        """
        Indicate if RoI regressor produces per class regression deltas or not

        Returns:
            bool: `True` if regression deltas apply to all classes, `False`
                if per class regression deltas are computed
        """
        return False
