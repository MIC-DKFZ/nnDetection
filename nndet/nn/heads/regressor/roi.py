import functools
from abc import abstractmethod
from typing import Optional, Sequence

import torch
from torch import Tensor

from nndet.losses import GIoULoss, SmoothL1Loss
from nndet.nn.conv import nd_pool
from nndet.nn.heads.abstract import Regressor, RoIConv1x1View


class RoIRegressor(Regressor):
    def __init__(
        self,
        conv,
        input_size: Sequence[int],
        in_channels: int,
        internal_channels: int,
        num_convs: int = 1,
        add_norm: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.dim = conv.dim
        self.num_convs = num_convs

        self.in_channels = in_channels
        self.internal_channels = internal_channels
        self.input_size = input_size

        self.module_internal = self._build_module_internal(
            conv=conv, add_norm=add_norm, **kwargs
        )
        self.module_out = self._build_module_out(conv=conv)

        self.loss: Optional[torch.nn.Module] = None
        self.init_weights()

    @abstractmethod
    def _build_module_internal(self, conv, add_norm: bool, **kwargs):
        raise NotImplementedError

    def _build_module_out(self, conv):
        return conv(
            self.internal_channels,
            self.dim * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            add_norm=False,
            add_act=False,
            bias=True,
        )

    def init_weights(self):
        pass

    def forward(self, features):
        x = self.module_out(self.module_internal(features))
        return x.view(x.shape[0], -1)  # [N, C, 1] -> [N, C]

    def compute_loss(
        self,
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


class ConvRoIRegressor(RoIRegressor):
    def _build_module_internal(self, conv, **kwargs):
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
    def _build_module_internal(self, conv, **kwargs):
        _conv_internal = torch.nn.Sequential()
        _conv_internal.add_module(
            name="conv1x1_view",
            module=RoIConv1x1View(dim=self.dim),
        )
        _conv_internal.add_module(
            name="c_in",
            module=conv(
                self.in_channels
                * functools.reduce(lambda a, b: a * b, self.input_size),
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


class L1ConvRoIRegressor(ConvRoIRegressor):
    def __init__(
        self,
        conv,
        in_channels: int,
        internal_channels: int,
        input_size: Sequence[int],
        num_convs: int = 1,
        add_norm: bool = True,
        beta: float = 1.0,
        reduction: Optional[str] = "sum",
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ):
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            internal_channels=internal_channels,
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


class GIoUConvRoIRegressor(ConvRoIRegressor):
    def __init__(
        self,
        conv,
        in_channels: int,
        internal_channels: int,
        input_size: Sequence[int],
        num_convs: int = 1,
        add_norm: bool = True,
        reduction: Optional[str] = "sum",
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ):
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            internal_channels=internal_channels,
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


class L1FCRoIRegressor(FCRoIRegressor):
    def __init__(
        self,
        conv,
        in_channels: int,
        internal_channels: int,
        input_size: Sequence[int],
        num_convs: int = 1,
        add_norm: bool = True,
        beta: float = 1.0,
        reduction: Optional[str] = "sum",
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ):
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            internal_channels=internal_channels,
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


class GIoUFCRoIRegressor(FCRoIRegressor):
    def __init__(
        self,
        conv,
        in_channels: int,
        internal_channels: int,
        input_size: Sequence[int],
        num_convs: int = 1,
        add_norm: bool = True,
        reduction: Optional[str] = "sum",
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ):
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            internal_channels=internal_channels,
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
