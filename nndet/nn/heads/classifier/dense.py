# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor

from nndet.losses.classification.ce import BCELoss, CELoss
from nndet.losses.classification.focal import AsymmetricBFocalLoss, BFocalLoss
from nndet.losses.classification.poly1 import Poly1BCEWithLogits, Poly1BFocalLoss
from nndet.nn.heads.abstract import Classifier
from nndet.utils.collections import CONV_TYPES


class DenseClassifier(Classifier):
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
        **kwargs,
    ):
        """
        Base class to build classifier heads with typical conv structure
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            num_classes: number of foreground classes
            anchors_per_pos: number of anchors per position
            num_levels: number of decoder levels which are passed through the
                classifier
            num_convs: number of convolutions
                input_conv -> num_convs -> output_convs
            add_norm: en-/disable normalization layers in internal layers
            kwargs: keyword arguments passed to first and internal convolutions

        Notes:
            `self.loss` needs to be overwritten in subclasses
            `self.logits_convert_fn` needs to be overwritten in subclasses
        """
        super().__init__()
        self.dim = conv.dim
        self.num_levels = num_levels
        self.num_convs = num_convs

        self.num_classes = num_classes
        self.anchors_per_pos = anchors_per_pos

        self.in_channels = in_channels
        self.internal_channels = internal_channels

        self.conv_internal = self.build_conv_internal(conv, add_norm=add_norm, **kwargs)
        self.conv_out = self.build_conv_out(conv)

        self.loss: Optional[nn.Module] = None
        self.logits_convert_fn: Optional[nn.Module] = None
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
        out_channels = self.num_classes * self.anchors_per_pos
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

    def forward(
        self,
        x: torch.Tensor,
        level: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward input

        Args:
            x: input feature map of size [N, C, dims], where N=batch size,
                C=number of channels, dims=spatial dimensions
            level: ignored, kept for compatibility with regressor head

        Returns:
            torch.Tensor: predicted logits [N, anchors, num_classes],
                where N=number batch_size, anchors=product of spatial size of
                feature map times number of anchors per position,
                num_classes=number of foreground classes (if softmax
                based predictions are used, one additional background channel
                at the 0th position is added)
        """
        class_logits = self.conv_out(self.conv_internal(x))

        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        class_logits = class_logits.permute(*axes)
        class_logits = class_logits.contiguous()
        class_logits = class_logits.view(x.size()[0], -1, self.num_classes)
        return class_logits

    def compute_loss(self, pred_logits: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """
        Compute loss from logits and targets with specified loss function
        (defined by `self.loss`).

        Args:
            pred_logits: predicted logits [N, C] where N=number of anchors,
                C=number of classes
            targets: classification targets [N], where N=number of anchors
                (targets need to be provided in numerical format as
                expected by CE loss from torch), (0 is background)

        Returns:
            Tensor: classification loss (scalar)
        """
        return self.loss(pred_logits, targets, **kwargs)

    def logits_to_probs(self, logits: Tensor) -> Tensor:
        """
        Convert bounding box logits to probabilities

        Args:
            logits: predicted logits [N, C]
                N = number of anchors, C=number of foreground classes

        Returns:
            Tensor: probabilities [N, C] where N = number of anchors,
                C=number of foreground classes
        """
        return self.logits_convert_fn(logits)

    def init_weights(self) -> None:
        """
        Init weights with prior prob
        """
        if self.prior_prob is not None:
            logger.info(f"Init classifier weights: prior prob {self.prior_prob}")
            for layer in self.modules():
                if isinstance(layer, CONV_TYPES):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)

            # Use prior in model initialization to improve stability
            bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
            for layer in self.conv_out.modules():
                if isinstance(layer, CONV_TYPES):
                    torch.nn.init.constant_(layer.bias, bias_value)
        else:
            logger.info("Init classifier weights: conv default")


class BCECLassifier(DenseClassifier):
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
        prior_prob: Optional[float] = None,
        weight: Optional[Tensor] = None,
        reduction: str = "mean",
        smoothing: float = 0.0,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ):
        """
        Classifier Head with sigmoid based BCE loss computation and prio
        prob weight init
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            num_classes: number of foreground classes
            anchors_per_pos: number of anchors per position
            num_levels: number of decoder levels which are passed through the
                classifier
            num_convs: number of convolutions
                input_conv -> num_convs -> output_convs
            add_norm: en-/disable normalization layers in internal layers
            prior_prob: initialize final conv with given prior probability
            weight: weight in BCEWithLogitsLoss (see pytorch for more info)
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            smoothing:  label smoothing
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            kwargs: keyword arguments passed to first and internal convolutions
        """
        self.prior_prob = prior_prob
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            num_convs=num_convs,
            add_norm=add_norm,
            internal_channels=internal_channels,
            num_classes=num_classes,
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            **kwargs,
        )

        self.loss = BCELoss(
            weight=weight,
            reduction=reduction,
            smoothing=smoothing,
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
        )
        self.logits_convert_fn = nn.Sigmoid()


class CEClassifier(DenseClassifier):
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
        prior_prob: Optional[float] = None,
        weight: Optional[Tensor] = None,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ):
        """
        Classifier Head with softmax based CE loss computation and prio
        prob weight init
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            num_classes: number of foreground classes
            anchors_per_pos: number of anchors per position
            num_levels: number of decoder levels which are passed through the
                classifier
            num_convs: number of convolutions
                input_conv -> num_convs -> output_convs
            add_norm: en-/disable normalization layers in internal layers
            prior_prob: initialize final conv with given prior probability
            weight: weight in cross entrpoy loss (see pytorch for more info)
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            kwargs: keyword arguments passed to first and internal convolutions
        """
        self.prior_prob = prior_prob
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            num_convs=num_convs,
            add_norm=add_norm,
            internal_channels=internal_channels,
            num_classes=num_classes + 1,  # add one channel for background
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            **kwargs,
        )

        self.loss = CELoss(
            weight=weight,
            reduction=reduction,
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
        )
        self.logits_convert_fn = nn.Softmax(dim=1)

    def logits_to_probs(self, logits: Tensor) -> Tensor:
        """
        Convert bounding box logits to probabilities

        Args:
            logits: predicted logits [N, C + 1]
                N = number of anchors, C=number of foreground classes

        Returns:
            Tensor: probabilities [N, C] where N = number of anchors,
                C=number of foreground classes
        """
        return self.logits_convert_fn(logits)[:, 1:]  # remove background predictions


class FocalClassifier(DenseClassifier):
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
        prior_prob: Optional[float] = None,
        gamma: float = 2,
        alpha: float = -1,
        reduction: str = "sum",
        smoothing: float = 0.0,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ):
        """
        Classifier Head with sigmoid based Focal loss computation and
        prio prob weight init
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            num_classes: number of foreground classes
            anchors_per_pos: number of anchors per position
            num_levels: number of decoder levels which are passed through the
                classifier
            num_convs: number of convolutions
                input_conv -> num_convs -> output_convs
            add_norm: en-/disable normalization layers in internal layers
            prior_prob: initialize final conv with given prior probability
            gamma: focal loss gamma
            alpha: focal loss alpha
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            smoothing:  label smoothing
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            kwargs: keyword arguments passed to first and internal convolutions
        """
        self.prior_prob = prior_prob
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            num_convs=num_convs,
            add_norm=add_norm,
            internal_channels=internal_channels,
            num_classes=num_classes,
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            **kwargs,
        )

        self.loss = BFocalLoss(
            gamma=gamma,
            alpha=alpha,
            loss_fp32=loss_fp32,
            loss_weight=loss_weight,
            reduction=reduction,
            smoothing=smoothing,
        )
        self.logits_convert_fn = nn.Sigmoid()


class AsymmetricFocalClassifier(FocalClassifier):
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
        prior_prob: Optional[float] = None,
        gamma: float = 2,
        alpha: float = -1,
        reduction: str = "sum",
        smoothing: float = 0.0,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ):
        """
        Classifier Head with sigmoid based Asym Focal loss computation and
        prio prob weight init
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            num_classes: number of foreground classes
            anchors_per_pos: number of anchors per position
            num_levels: number of decoder levels which are passed through the
                classifier
            num_convs: number of convolutions
                input_conv -> num_convs -> output_convs
            add_norm: en-/disable normalization layers in internal layers
            prior_prob: initialize final conv with given prior probability
            gamma: focal loss gamma
            alpha: focal loss alpha
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            smoothing:  label smoothing
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            kwargs: keyword arguments passed to first and internal convolutions
        """
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            num_convs=num_convs,
            add_norm=add_norm,
            internal_channels=internal_channels,
            num_classes=num_classes,
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            prior_prob=prior_prob,
            **kwargs,
        )

        self.loss = AsymmetricBFocalLoss(
            gamma=gamma,
            alpha=alpha,
            loss_fp32=loss_fp32,
            loss_weight=loss_weight,
            reduction=reduction,
            smoothing=smoothing,
        )
        self.logits_convert_fn = nn.Sigmoid()


class Poly1BCECLassifier(DenseClassifier):
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
        prior_prob: Optional[float] = None,
        alpha: float = -1,
        epsilon: float = -1,
        reduction: str = "mean",
        smoothing: float = 0.0,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ):
        """
        Classifier Head with sigmoid based Poly1 BCE loss computation and prio
        prob weight init
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            num_classes: number of foreground classes
            anchors_per_pos: number of anchors per position
            num_levels: number of decoder levels which are passed through the
                classifier
            num_convs: number of convolutions
                input_conv -> num_convs -> output_convs
            add_norm: en-/disable normalization layers in internal layers
            prior_prob: initialize final conv with given prior probability
            alpha: balance positive and negative samples [0, 1] (increasing
                alpha increase weight of foreground classes (better recall))
            epsilon: epsilon of poly term.
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            smoothing:  label smoothing
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            kwargs: keyword arguments passed to first and internal convolutions
        """
        self.prior_prob = prior_prob
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            num_convs=num_convs,
            add_norm=add_norm,
            internal_channels=internal_channels,
            num_classes=num_classes,
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            **kwargs,
        )
        self.loss = Poly1BCEWithLogits(
            alpha=alpha,
            epsilon=epsilon,
            loss_fp32=loss_fp32,
            loss_weight=loss_weight,
            reduction=reduction,
            smoothing=smoothing,
        )
        self.logits_convert_fn = nn.Sigmoid()


class Poly1FocalClassifier(DenseClassifier):
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
        prior_prob: Optional[float] = None,
        gamma: float = 2,
        alpha: float = -1,
        epsilon: float = -1,
        reduction: str = "sum",
        smoothing: float = 0.0,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ):
        """
        Classifier Head with sigmoid based Poly1 Focal loss computation and
        prio prob weight init
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            num_classes: number of foreground classes
            anchors_per_pos: number of anchors per position
            num_levels: number of decoder levels which are passed through the
                classifier
            num_convs: number of convolutions
                input_conv -> num_convs -> output_convs
            add_norm: en-/disable normalization layers in internal layers
            prior_prob: initialize final conv with given prior probability
            gamma: focal loss gamma
            alpha: focal loss alpha
            epsilon: epsilon of poly term.
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            smoothing:  label smoothing
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            kwargs: keyword arguments passed to first and internal convolutions
        """
        self.prior_prob = prior_prob
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            num_convs=num_convs,
            add_norm=add_norm,
            internal_channels=internal_channels,
            num_classes=num_classes,
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            **kwargs,
        )

        self.loss = Poly1BFocalLoss(
            gamma=gamma,
            alpha=alpha,
            epsilon=epsilon,
            loss_fp32=loss_fp32,
            loss_weight=loss_weight,
            reduction=reduction,
            smoothing=smoothing,
        )
        self.logits_convert_fn = nn.Sigmoid()


class FullyConntectedBCECLassifier(BCECLassifier):
    """
    BCE Classifier with 1x1 convs which act as fc
    layers with shared weights across spatial locations

    conv3(in, internal) -> num_convs x conv1(internal, internal) -> conv1(internal, out)
    """

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
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    **kwargs,
                ),
            )
        return _conv_internal

    def build_conv_out(self, conv):
        """
        Build final convolutions
        """
        out_channels = self.num_classes * self.anchors_per_pos
        return conv(
            self.internal_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            add_norm=False,
            add_act=False,
            bias=True,
        )
