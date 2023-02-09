# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import functools
import math
from abc import abstractmethod
from typing import Optional, Sequence

import torch
from loguru import logger
from torch import Tensor

from nndet.losses.classification.ce import BCELoss, CELoss
from nndet.nn.heads.abstract import Classifier, RoIConv1x1View
from nndet.nn.layers.wrapper import nd_pool
from nndet.utils.collections import CONV_TYPES


class RoIClassifier(Classifier):
    def __init__(
        self,
        conv,
        input_size: Sequence[int],
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        num_convs: int = 1,
        add_norm: bool = True,
        **kwargs,
    ):
        """
        Base class to build RoI classifier heads with typical conv structure
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            num_classes: number of foreground classes
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
        self.num_convs = num_convs
        self.num_classes = num_classes

        self.in_channels = in_channels
        self.internal_channels = internal_channels
        self.input_size = input_size

        self.module_internal = self._build_module_internal(conv=conv, add_norm=add_norm, **kwargs)
        self.module_out = self._build_module_out(conv=conv)

        self.loss: Optional[torch.nn.Module] = None
        self.logits_convert_fn: Optional[torch.nn.Module] = None
        self.init_weights()

    @abstractmethod
    def _build_module_internal(self, conv, add_norm: bool, **kwargs):
        """
        Build internal modules
        """
        raise NotImplementedError

    def _build_module_out(self, conv):
        """
        Build final convolution
        """
        return conv(
            self.internal_channels,
            self.num_classes,
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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward input through module

        Args:
            features: input features [N, C, roi_size], where N=number of
                RoIs, C=number of channels, roi_size=spatial size of RoI

        Returns:
            torch.Tensor: predicted logits [N, num_classes], where N=number of
                RoIs, num_classes=number of foreground classes (if softmax
                based predictions are used, one additional background channel
                at the 0th position is added)
        """
        x = self.module_out(self.module_internal(features))
        return x.view(x.shape[0], -1)  # [N, C, 1] -> [N, C]

    def compute_loss(self, pred_logits: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """
        Compute loss from logits and targets with specified loss function
        (defined by `self.loss`).

        Args:
            pred_logits: predicted logits [N, C] where N=number of RoIs,
                C=number of classes
            targets: classification targets [N], where N=number of RoIs
                (targets need to be provided in numerical format as
                expected by CE loss from torch)

        Returns:
            Tensor: classification loss (scalar)
        """
        return self.loss(pred_logits, targets, **kwargs)

    def logits_to_probs(self, logits: Tensor) -> Tensor:
        """
        Convert bounding box logits to probabilities

        Args:
            logits: predicted logits [N, C]
                N = number of RoIs, C=number of foreground classes (if
                softmax based predicitions are used, num_classes + 1
                channels will be available in the input)

        Returns:
            Tensor: probabilities [N, C] where N = number of RoIs,
                C=number of foreground classes
        """
        return self.logits_convert_fn(logits)


class ConvRoIClassifier(RoIClassifier):
    def _build_module_internal(self, conv, **kwargs):
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


class FCRoIClassifier(RoIClassifier):
    def _build_module_internal(self, conv, **kwargs):
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


class BCEConvRoIClassifier(ConvRoIClassifier):
    def __init__(
        self,
        conv,
        input_size: Sequence[int],
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        num_convs: int = 1,
        add_norm: bool = True,
        weight: Optional[Tensor] = None,
        reduction: str = "sum",
        smoothing: float = 0.0,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        prior_prob: Optional[float] = None,
        **kwargs,
    ):
        """
        Classifier Head with sigmoid based BCE loss computation and prio
        prob weight init. Structure: conv(s) -> pool -> out

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            num_classes: number of foreground classes
            num_convs: number of convolutions
                input_conv -> num_convs -> output_convs
            add_norm: en-/disable normalization layers in internal layers
            weight: weight in BCEWithLogitsLoss (see pytorch for more info)
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            smoothing:  label smoothing
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            prior_prob: initialize final conv with given prior probability.
                If `None`, no init will be performed.
            kwargs: keyword arguments passed to first and internal convolutions
        """
        self.prior_prob = prior_prob
        super().__init__(
            conv=conv,
            input_size=input_size,
            in_channels=in_channels,
            num_convs=num_convs,
            add_norm=add_norm,
            internal_channels=internal_channels,
            num_classes=num_classes,
            **kwargs,
        )
        self.loss = BCELoss(
            weight=weight,
            reduction=reduction,
            smoothing=smoothing,
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
        )
        self.logits_convert_fn = torch.nn.Sigmoid()

    def init_weights(self) -> None:
        """
        Init weights with prior prob
        """
        if self.prior_prob is not None:
            logger.info(f"Init RoI classifier weights: prior prob {self.prior_prob}")
            for layer in self.modules():
                if isinstance(layer, CONV_TYPES):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)

            # Use prior in model initialization to improve stability
            if math.isclose(self.prior_prob, 0):
                logger.info("Found prior prob 0, init bias with 0")
                bias_value = 0
            else:
                bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)

            for layer in self.module_out.modules():
                if isinstance(layer, CONV_TYPES):
                    torch.nn.init.constant_(layer.bias, bias_value)
        else:
            logger.info("Init RoI classifier weights: conv default")


class BCEFCRoIClassifier(FCRoIClassifier):
    def __init__(
        self,
        conv,
        input_size: Sequence[int],
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        num_convs: int = 1,
        add_norm: bool = True,
        weight: Optional[Tensor] = None,
        reduction: str = "sum",
        smoothing: float = 0.0,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        prior_prob: Optional[float] = None,
        **kwargs,
    ):
        """
        Classifier Head with sigmoid based BCE loss computation and prio
        prob weight init. Structure: flatten -> FC(s) -> out

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            num_classes: number of foreground classes
            num_convs: number of convolutions
                input_conv -> num_convs -> output_convs
            add_norm: en-/disable normalization layers in internal layers
            weight: weight in BCEWithLogitsLoss (see pytorch for more info)
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            smoothing:  label smoothing
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            prior_prob: initialize final conv with given prior probability.
                If `None`, no init will be performed.
            kwargs: keyword arguments passed to first and internal convolutions
        """
        self.prior_prob = prior_prob
        super().__init__(
            conv=conv,
            input_size=input_size,
            in_channels=in_channels,
            num_convs=num_convs,
            add_norm=add_norm,
            internal_channels=internal_channels,
            num_classes=num_classes,
            **kwargs,
        )
        self.loss = BCELoss(
            weight=weight,
            reduction=reduction,
            smoothing=smoothing,
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
        )
        self.logits_convert_fn = torch.nn.Sigmoid()

    def init_weights(self) -> None:
        """
        Init weights with prior prob
        """
        if self.prior_prob is not None:
            logger.info(f"Init RoI classifier weights: prior prob {self.prior_prob}")
            for layer in self.modules():
                if isinstance(layer, CONV_TYPES):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)

            # Use prior in model initialization to improve stability
            if math.isclose(self.prior_prob, 0):
                logger.info("Found prior prob 0, init bias with 0")
                bias_value = 0
            else:
                bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)

            for layer in self.module_out.modules():
                if isinstance(layer, CONV_TYPES):
                    torch.nn.init.constant_(layer.bias, bias_value)
        else:
            logger.info("Init RoI classifier weights: conv default")


class CEConvRoIClassifier(ConvRoIClassifier):
    def __init__(
        self,
        conv,
        input_size: Sequence[int],
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        num_convs: int = 1,
        add_norm: bool = True,
        weight: Optional[Tensor] = None,
        reduction: str = "sum",
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ):
        """
        Classifier Head with softmax based CE loss computation.
        Structure: conv(s) -> pool -> out

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            num_classes: number of foreground classes
            num_convs: number of convolutions
                input_conv -> num_convs -> output_convs
            add_norm: en-/disable normalization layers in internal layers
            weight: weight in BCEWithLogitsLoss (see pytorch for more info)
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            kwargs: keyword arguments passed to first and internal convolutions
        """
        super().__init__(
            conv=conv,
            input_size=input_size,
            in_channels=in_channels,
            num_convs=num_convs,
            add_norm=add_norm,
            internal_channels=internal_channels,
            num_classes=num_classes + 1,  # add one channel for background
            **kwargs,
        )
        self.loss = CELoss(
            weight=weight,
            reduction=reduction,
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
        )
        self.logits_convert_fn = torch.nn.Softmax(dim=1)

    def logits_to_probs(self, logits: Tensor) -> Tensor:
        """
        Convert bounding box logits to probabilities

        Args:
            logits: predicted logits [N, C]
                N = number of RoIs, C=number of foreground classes + 1
                    (+1 needed for background in softmax)

        Returns:
            Tensor: probabilities [N, C] where N = number of RoIs,
                C=number of foreground classes
        """
        return self.logits_convert_fn(logits)[:, 1:]  # remove background predictions


class CEFCRoIClassifier(FCRoIClassifier):
    def __init__(
        self,
        conv,
        input_size: Sequence[int],
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        num_convs: int = 1,
        add_norm: bool = True,
        weight: Optional[Tensor] = None,
        reduction: str = "sum",
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ):
        """
        Classifier Head with softmax based CE loss computation.
        Structure: flatten -> FC(s) -> out

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            num_classes: number of foreground classes
            num_convs: number of convolutions
                input_conv -> num_convs -> output_convs
            add_norm: en-/disable normalization layers in internal layers
            weight: weight in BCEWithLogitsLoss (see pytorch for more info)
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            kwargs: keyword arguments passed to first and internal convolutions
        """
        super().__init__(
            conv=conv,
            input_size=input_size,
            in_channels=in_channels,
            num_convs=num_convs,
            add_norm=add_norm,
            internal_channels=internal_channels,
            num_classes=num_classes + 1,  # add one channel for background
            **kwargs,
        )
        self.loss = CELoss(
            weight=weight,
            reduction=reduction,
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
        )
        self.logits_convert_fn = torch.nn.Softmax(dim=1)

    def logits_to_probs(self, logits: Tensor) -> Tensor:
        """
        Convert bounding box logits to probabilities

        Args:
            logits: predicted logits [N, C]
                N = number of RoIs, C=number of foreground classes + 1
                    (+1 needed for background in softmax)

        Returns:
            Tensor: probabilities [N, C] where N = number of RoIs,
                C=number of foreground classes
        """
        return self.logits_convert_fn(logits)[:, 1:]  # remove background predictions
