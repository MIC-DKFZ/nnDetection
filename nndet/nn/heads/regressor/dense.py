# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor

from nndet.losses.regression.diou import DIoULoss
from nndet.losses.regression.giou import GIoULoss, GIoULossPaired
from nndet.losses.regression.smoothl1 import SmoothL1Loss
from nndet.nn.heads.abstract import CONV_TYPES, Regressor
from nndet.nn.ops.scale import Scale, ScalePerDim


class DenseRegressor(Regressor):
    def __init__(
        self,
        conv,
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        anchors_per_pos: int,
        num_levels: int,
        num_convs: int = 3,
        add_norm: bool = True,
        learn_scale: bool = False,
        scale_per_dim: bool = False,
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
            num_classes: number of foreground classes
            anchors_per_pos: number of anchors per position
            num_levels: number of decoder levels which are passed through the
                regressor
            num_convs: number of convolutions
                in conv -> num convs -> final conv
            add_norm: en-/disable normalization layers in internal layers
            learn_scale: learn additional single scalar values per feature
                pyramid level
            scale_per_dim: if `learn_scale` is `True`, the scale is learned
                for each spatial dimension separately
            kwargs: keyword arguments passed to first and internal convolutions
        """
        super().__init__()
        self.dim = conv.dim
        self.num_levels = num_levels
        self.num_convs = num_convs
        self.num_classes = num_classes
        self.learn_scale = learn_scale
        self.scale_per_dim = scale_per_dim

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
        logger.info(f"Learning level specific scalar in regressor with scale per dim {self.scale_per_dim}")
        if self.scale_per_dim:
            scale = [1.0 for _ in range(self.dim)]
            return nn.ModuleList([ScalePerDim(scale=scale) for _ in range(self.num_levels)])
        else:
            return nn.ModuleList([Scale() for _ in range(self.num_levels)])

    def forward(self, x: torch.Tensor, level: int, **kwargs) -> torch.Tensor:
        """
        Forward input

        Args:
            x: input feature map of size [N, C, dims], where N=batch size,
                C=number of channels, dims=spatial dimensions
            level: specify current level, needed if level specific
                modules should be forward

        Returns:
            torch.Tensor: predicted logits [N, anchors, dim*2],
                where N=number batch_size, anchors=product of spatial size of
                feature map times number of anchors per position,
                dim=number of spatial dimensions
        """
        bb_logits = self.conv_out(self.conv_internal(x))

        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        bb_logits = bb_logits.permute(*axes)
        bb_logits = bb_logits.contiguous()
        bb_logits = bb_logits.view(x.size()[0], -1, self.dim * 2)

        if self.learn_scale:
            bb_logits = self.scales[level](bb_logits)

        return bb_logits

    def compute_loss(
        self,
        pred_deltas: Tensor,
        target_deltas: Tensor,
        target_labels: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Compute regression loss (l1 loss)

        Args:
            pred_deltas: predicted bounding box deltas [N,  dim * 2]
            target_deltas: target bounding box deltas [N,  dim * 2]
            target_labels: target labels for boxes [N], where
                N=number of anchors
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

    @classmethod
    def class_agnostic(cls):
        """
        All dense regressor should be class agnostic

        Returns:
            bool: `True` if regression deltas apply to all classes, `False`
                if per class regression deltas are computed
        """
        return True


class L1Regressor(DenseRegressor):
    def __init__(
        self,
        conv,
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        anchors_per_pos: int,
        num_levels: int,
        num_convs: int = 3,
        add_norm: bool = True,
        beta: float = 1.0,
        reduction: Optional[str] = "sum",
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        learn_scale: bool = False,
        scale_per_dim: bool = False,
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
            num_classes: number of foreground classes
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
            loss_fp32: If True, loss is forced to be computed in float32
            learn_scale: learn additional single scalar values per feature
                pyramid level
            scale_per_dim: if `learn_scale` is `True`, the scale is learned
                for each spatial dimension separately
            kwargs: keyword arguments passed to first and internal convolutions
        """
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            internal_channels=internal_channels,
            num_classes=num_classes,
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            num_convs=num_convs,
            add_norm=add_norm,
            learn_scale=learn_scale,
            scale_per_dim=scale_per_dim,
            **kwargs,
        )
        self.loss = SmoothL1Loss(
            beta=beta,
            reduction=reduction,
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
        )


class GIoURegressor(DenseRegressor):
    def __init__(
        self,
        conv,
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        anchors_per_pos: int,
        num_levels: int,
        num_convs: int = 3,
        add_norm: bool = True,
        reduction: Optional[str] = "sum",
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        learn_scale: bool = False,
        scale_per_dim: bool = False,
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
            num_classes: number of foreground classes
            anchors_per_pos: number of anchors per position
            num_levels: number of decoder levels which are passed through the
                regressor
            num_convs: number of convolutions
                in conv -> num convs -> final conv
            add_norm: en-/disable normalization layers in internal layers
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            loss_weight: scalar to balance multiple losses
            loss_fp32: IGNORED, loss is always computed in fp32. This argument
                is only added here to have a uniform API.
            learn_scale: learn additional single scalar values per feature
                pyramid level
            scale_per_dim: if `learn_scale` is `True`, the scale is learned
                for each spatial dimension separately
            kwargs: keyword arguments passed to first and internal convolutions
        """
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            internal_channels=internal_channels,
            num_classes=num_classes,
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            num_convs=num_convs,
            add_norm=add_norm,
            learn_scale=learn_scale,
            scale_per_dim=scale_per_dim,
            **kwargs,
        )
        self.loss = GIoULoss(
            reduction=reduction,
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
        )


class GIoUPRegressor(DenseRegressor):
    def __init__(
        self,
        conv,
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        anchors_per_pos: int,
        num_levels: int,
        num_convs: int = 3,
        add_norm: bool = True,
        reduction: Optional[str] = "sum",
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        learn_scale: bool = False,
        scale_per_dim: bool = False,
        **kwargs,
    ):
        """
        Build regressor heads with typical conv structure and generalized
        IoU loss
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        (only compute GIoU on paired bounding boxes, this should be more
        efficient than the previous regressor implemenetation)

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            num_classes: number of foreground classes
            anchors_per_pos: number of anchors per position
            num_levels: number of decoder levels which are passed through the
                regressor
            num_convs: number of convolutions
                in conv -> num convs -> final conv
            add_norm: en-/disable normalization layers in internal layers
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            loss_weight: scalar to balance multiple losses
            loss_fp32: IGNORED, loss is always computed in fp32. This argument
                is only added here to have a uniform API.
            learn_scale: learn additional single scalar values per feature
                pyramid level
            scale_per_dim: if `learn_scale` is `True`, the scale is learned
                for each spatial dimension separately
            kwargs: keyword arguments passed to first and internal convolutions
        """
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            internal_channels=internal_channels,
            num_classes=num_classes,
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            num_convs=num_convs,
            add_norm=add_norm,
            learn_scale=learn_scale,
            scale_per_dim=scale_per_dim,
            **kwargs,
        )
        self.loss = GIoULossPaired(
            reduction=reduction,
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
        )


class DIoURegressor(DenseRegressor):
    def __init__(
        self,
        conv,
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        anchors_per_pos: int,
        num_levels: int,
        num_convs: int = 3,
        add_norm: bool = True,
        reduction: Optional[str] = "sum",
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        learn_scale: bool = False,
        scale_per_dim: bool = False,
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
            num_classes: number of foreground classes
            anchors_per_pos: number of anchors per position
            num_levels: number of decoder levels which are passed through the
                regressor
            num_convs: number of convolutions
                in conv -> num convs -> final conv
            add_norm: en-/disable normalization layers in internal layers
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            loss_weight: scalar to balance multiple losses
            loss_fp32: IGNORED, loss is always computed in fp32. This argument
                is only added here to have a uniform API.
            learn_scale: learn additional single scalar values per feature
                pyramid level
            scale_per_dim: if `learn_scale` is `True`, the scale is learned
                for each spatial dimension separately
            kwargs: keyword arguments passed to first and internal convolutions
        """
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            internal_channels=internal_channels,
            num_classes=num_classes,
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            num_convs=num_convs,
            add_norm=add_norm,
            learn_scale=learn_scale,
            scale_per_dim=scale_per_dim,
            **kwargs,
        )
        self.loss = DIoULoss(
            reduction=reduction,
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
        )


class DualRegressor(DenseRegressor):
    def __init__(
        self,
        conv,
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        anchors_per_pos: int,
        num_levels: int,
        num_convs: int = 3,
        add_norm: bool = True,
        reduction: Optional[str] = "sum",
        beta: float = 1.0,
        loss_weight_l1: float = 5.0,
        loss_weight_giou: float = 2.0,
        loss_fp32: bool = False,
        learn_scale: bool = False,
        scale_per_dim: bool = False,
        **kwargs,
    ):
        """
        Build regressor heads with typical conv structure and GIoU and L1
        loss function: loss_weight * [(1-alpha) * L1 + alpha * GIoU]
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            num_classes: number of foreground classes
            anchors_per_pos: number of anchors per position
            num_levels: number of decoder levels which are passed through the
                regressor
            num_convs: number of convolutions
                in conv -> num convs -> final conv
            add_norm: en-/disable normalization layers in internal layers
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            beta: L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            alpha: balance loss functions
            loss_weight_l1: loss weight to balance l1 loss
            loss_weight_giou: loss weight to balance giou loss
            loss_fp32: If True, l1 loss is forced to be computed in float32
            learn_scale: learn additional single scalar values per feature
                pyramid level
            scale_per_dim: if `learn_scale` is `True`, the scale is learned
                for each spatial dimension separately
            kwargs: keyword arguments passed to first and internal convolutions
        """
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            internal_channels=internal_channels,
            num_classes=num_classes,
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            num_convs=num_convs,
            add_norm=add_norm,
            learn_scale=learn_scale,
            scale_per_dim=scale_per_dim,
            **kwargs,
        )
        self.loss_weight_l1 = loss_weight_l1
        self.loss_weight_giou = loss_weight_giou
        self.loss_l1 = SmoothL1Loss(
            beta=beta,
            reduction=reduction,
            loss_fp32=loss_fp32,
        )
        self.loss_giou = GIoULoss(
            reduction=reduction,
            loss_fp32=loss_fp32,
        )

    def compute_loss(
        self,
        pred_deltas: Tensor,
        target_deltas: Tensor,
        pred_boxes: Tensor,
        target_boxes: Tensor,
        target_labels: torch.Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Compute regression loss (l1 loss)

        Args:
            pred_deltas: predicted bounding box deltas [N,  dim * 2]
            target_deltas: target bounding box deltas [N,  dim * 2]
            pred_boxes: predicted bounding boxes [N,  dim * 2]
            target_boxes: target bounding boxes [N,  dim * 2]
            target_labels: target labels for boxes [N], where
                N=number of anchors
            kwargs: ignored

        Returns:
            Tensor: loss
        """
        if not self.class_agnostic():
            # only compute loss on target class
            num_rois, _ = pred_deltas.shape
            _target_labels = target_labels - 1  # matching adds +1 for background which needs to be removed

            _pred_deltas = pred_deltas.reshape(num_rois, self.num_classes, self.dim * 2)
            _pred_deltas = _pred_deltas[torch.arange(num_rois), _target_labels]

            _pred_boxes = pred_boxes.reshape(num_rois, self.num_classes, self.dim * 2)
            _pred_boxes = _pred_boxes[torch.arange(num_rois), _target_labels]
        else:
            _pred_deltas = pred_deltas
        l1 = self.loss_l1(_pred_deltas, target_deltas)
        giou = self.loss_giou(_pred_boxes, target_boxes)
        return l1 * self.loss_weight_l1 + giou * self.loss_weight_giou
