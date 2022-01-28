import functools
from abc import abstractmethod
from typing import Optional, Sequence

import torch
from torch import Tensor

from nndet.arch.conv import nd_pool
from nndet.arch.heads.abstract import Classifier, RoIConv1x1View
from nndet.losses.classification import BCEWithLogitsLossOneHot, CrossEntropyLoss


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
        super().__init__()
        self.dim = conv.dim
        self.num_convs = num_convs
        self.num_classes = num_classes

        self.in_channels = in_channels
        self.internal_channels = internal_channels
        self.input_size = input_size

        self.module_internal = self._build_module_internal(
            conv=conv, add_norm=add_norm, **kwargs
        )
        self.module_out = self._build_module_out(conv=conv)

        self.loss: Optional[torch.nn.Module] = None
        self.logits_convert_fn: Optional[torch.nn.Module] = None
        self.init_weights()

    @abstractmethod
    def _build_module_internal(self, conv, add_norm: bool, **kwargs):
        raise NotImplementedError

    def _build_module_out(self, conv):
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
        pass

    def forward(self, features):
        x = self.module_out(self.module_internal(features))
        return x.view(x.shape[0], -1)  # [N, C, 1] -> [N, C]

    def compute_loss(self, pred_logits: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """
        Base classifier with cross entropy loss (in general hard negative
        example mining should be done before this)

        Args:
            pred_logits (Tensor): predicted logits
            targets (Tensor): classification targets

        Returns:
            Tensor: classification loss
        """
        return self.loss(pred_logits, targets, **kwargs)

    def logits_to_probs(self, logits: Tensor) -> Tensor:
        """
        Convert bounding box logits to probabilities

        Args:
            logits (Tensor): bounding box logits [N, C]
                N = number of anchors, C=number of foreground classes

        Returns:
            Tensor: probabilities
        """
        return self.logits_convert_fn(logits)


class ConvRoIClassifier(RoIClassifier):
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


class FCRoIClassifier(RoIClassifier):
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
        **kwargs,
    ):
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
        self.loss = BCEWithLogitsLossOneHot(
            num_classes=num_classes,
            weight=weight,
            reduction=reduction,
            smoothing=smoothing,
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
        )
        self.logits_convert_fn = torch.nn.Sigmoid()


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
        **kwargs,
    ):
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
        self.loss = BCEWithLogitsLossOneHot(
            num_classes=num_classes,
            weight=weight,
            reduction=reduction,
            smoothing=smoothing,
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
        )
        self.logits_convert_fn = torch.nn.Sigmoid()


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
        self.loss = CrossEntropyLoss(
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
            logits (Tensor): bounding box logits [N, C], C=number of classes

        Returns:
            Tensor: probabilities
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
        self.loss = CrossEntropyLoss(
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
            logits (Tensor): bounding box logits [N, C], C=number of classes

        Returns:
            Tensor: probabilities
        """
        return self.logits_convert_fn(logits)[:, 1:]  # remove background predictions
