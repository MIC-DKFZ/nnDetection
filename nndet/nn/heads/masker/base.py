# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import math
from abc import abstractmethod
from typing import Optional, TypeVar

import torch
from loguru import logger
from torch import Tensor, nn

from nndet.losses.mask.ce import BCEMaskLoss
from nndet.losses.mask.dice import BDiceMaskLoss
from nndet.nn.heads.abstract import Classifier
from nndet.utils.collections import CONV_TYPES

# TODO: cleanup


class Masker(Classifier):
    def __init__(
        self,
        conv,
        in_channels: int,
        internal_channels: int,
        num_convs: int = 3,
        add_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dim = conv.dim
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.internal_channels = internal_channels

        self.conv_internal = self.build_conv_internal(conv, add_norm=add_norm, **kwargs)
        self.conv_out = self.build_conv_out(conv)

        self.loss: Optional[nn.Module] = None
        self.logits_convert_fn: Optional[nn.Module] = None
        self.init_weights()

    @abstractmethod
    def get_output_channels(self) -> int:
        raise NotImplementedError

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
        _conv_out = nn.Sequential()
        _conv_out.add_module(
            name="c_transpose0",
            module=conv(
                self.internal_channels,
                self.internal_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                transposed=True,
                add_norm=False,
                add_act=True,
            ),
        )
        _conv_out.add_module(
            name="c_out",
            module=conv(
                self.internal_channels,
                self.get_output_channels(),
                kernel_size=1,
                stride=1,
                padding=0,
                add_norm=False,
                add_act=False,
                bias=True,
            ),
        )
        return _conv_out

    def init_weights(self):
        pass

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward input

        Args:
            x (torch.Tensor): input feature map of size (N x C x Y x X x Z)

        Returns:
            torch.Tensor: mask logits for each anchor
                [N, anchors, num_classes]
            torch.Tensor: mask features before upsampling
                [N, C, dims]
        """
        feat_out = self.conv_internal(x)
        return self.conv_out(feat_out), feat_out

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
        if pred_logits.numel() > 0:
            return {"mask": self.loss(pred_logits, targets, **kwargs)}
        else:
            return {"mask": pred_logits.new_zeros([1])}

    @abstractmethod
    def logits_to_probs(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        Convert bounding box logits to probabilities

        Args:
            logits: mask logits [N, C, dims], N=number of objects,
                C=number of classes, dims = spatial dimensions
            labels: predicted label for each mask [N]

        Returns:
            Tensor: probabilities
        """
        raise NotImplementedError

    @classmethod
    def class_agnostic(cls):
        return True


class BCESingleMasker(Masker):
    def __init__(self, *args, prior_prob: Optional[float] = None, **kwargs):
        self.prior_prob = prior_prob
        super().__init__(*args, **kwargs)
        self.loss = BCEMaskLoss()
        self.logits_convert_fn = torch.nn.Sigmoid()

    def get_output_channels(self) -> int:
        return 1

    def init_weights(self) -> None:
        """
        Init weights with prior prob
        """
        if self.prior_prob is not None:
            logger.info(f"Init RoI Masker weights: prior prob {self.prior_prob}")
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

            for layer in self.conv_out.modules():
                if isinstance(layer, CONV_TYPES):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.001)
                    torch.nn.init.constant_(layer.bias, bias_value)
        else:
            logger.info("Init RoI Masker weights: conv default")

    def logits_to_probs(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        Convert bounding box logits to probabilities

        Args:
            logits: mask logits [N, C, dims], N=number of objects,
                C=number of classes, dims = spatial dimensions
            labels: predicted label for each mask [N]

        Returns:
            Tensor: probabilities
        """
        return self.logits_convert_fn(logits).squeeze(dim=1)


class BDiceBCESingleMasker(BCESingleMasker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_dice = BDiceMaskLoss(
            batch_dice=False,
            smooth_nom=0,
            smooth_denom=1e-5,
            loss_weight=1.0,
            loss_fp32=True,
            reduction="mean",
        )

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
        if pred_logits.numel() > 0:
            mask_losses = {
                "mask_bce": self.loss(pred_logits, targets, **kwargs),
                "mask_dice": self.loss_dice(pred_logits, targets, **kwargs),
            }
        else:
            mask_losses = {
                "mask_bce": pred_logits.new_zeros([1]),
                "mask_dice": pred_logits.new_zeros([1]),
            }
        return mask_losses


MaskerType = TypeVar("MaskerType", bound=Masker)
