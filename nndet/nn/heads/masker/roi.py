# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import math
from abc import abstractclassmethod
from typing import Optional

import torch
from loguru import logger
from torch import Tensor, nn

from nndet.losses.mask.ce import BCEMaskLoss
from nndet.losses.mask.dice import BDiceMaskLoss
from nndet.nn.heads.abstract import Classifier
from nndet.utils.collections import CONV_TYPES
from nndet.utils.typing import CONVGEN, ND_TUPLE_INT


class Masker(Classifier):
    def __init__(
        self,
        conv: CONVGEN,
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        num_convs: int = 3,
        add_norm: bool = False,
        **kwargs,
    ):
        """
        Mask head of RCNN
        c_in -> c_internal x num_convs -> conv_transpose -> c_out

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of internal channels
            num_classes: number of foreground classes
            num_convs: Number of internal convolutions. Defaults to 3.
            add_norm: Add normalisation layers to internal convolutions.
                Defaults to False.
            kwargs: passed as keyword argument to in-/internal convolutions
        """
        super().__init__()
        self.dim = conv.dim
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.internal_channels = internal_channels
        self.num_classes = num_classes

        self.conv_internal = self.build_conv_internal(conv, add_norm=add_norm, **kwargs)
        self.conv_out = self.build_conv_out(conv)

        self.loss: Optional[nn.Module] = None
        self.loss_name: str = ""
        self.logits_convert_fn: Optional[nn.Module] = None
        self.init_weights()

    @abstractclassmethod
    def is_class_agnostic(cls) -> bool:
        """
        Indicate if mask head is class agnositic or class specific

        Returns:
            bool: `True` if predictions will be class agnostic, meaning that
                each RoI produces a mask with a single channel. If `False`
                each RoI produces a mask for each possible class. The loss
                is only computed on the matched label.
        """
        raise NotImplementedError

    def get_output_channels(self) -> int:
        """
        Retrieve number of ouptut channels

        Returns:
            int: number of output channels
        """
        return 1 if self.is_class_agnostic() else self.num_classes

    def get_upscale_factor(self) -> ND_TUPLE_INT:
        """
        Retrieve upscale factor of mask head

        Returns:
            ND_TUPLE_INT: upscale factor
        """
        return (2, 2) if self.dim == 2 else (2, 2, 2)

    def build_conv_internal(self, conv: CONVGEN, **kwargs) -> torch.nn.Module:
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

    def build_conv_out(self, conv: CONVGEN):
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
        """
        Initialize weights of head
        """
        pass

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward input

        Args:
            x: input feature map of size [N, C, *], where N is the number of
                RoIs, C is the number of input channels and * are
                2 or 3 spatial dimensions

        Returns:
            torch.Tensor: mask logits for each anchor
                [N, num_classes, dims], where N is the number of RoIs,
                num_classes is the number of foreground classes and dims
                are spatial dimensions
            torch.Tensor: mask features before upsampling
                [N, C, dims] where N is the number of RoIs,
                C is the number of internal channels and dims are spatial
                dimensions
        """
        feat_out = self.conv_internal(x)
        return self.conv_out(feat_out), feat_out

    def compute_loss(
        self,
        pred_logits: Tensor,
        target_masks: Tensor,
        target_labels: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Base classifier with cross entropy loss (in general hard negative
        example mining should be done before this)

        Args:
            pred_logits: predicted mask logits [N, num_classes, dims],
                where N is the number of RoIs, num_classes is the number of
                foreground classes and dims are spatial dimensions
            target_masks: mask targets as binary masks [N, dims]
                where N is the number of RoIs and dims are spatial dimensions
            target_labels: classification label for each mask [N]
                where N is the number of RoIs (0 is background)

        Returns:
            Tensor: loss
        """
        if pred_logits.numel() > 0:
            if not self.is_class_agnostic():
                roi_idx = torch.arange(pred_logits.shape[0])
                _target_labels = target_labels - 1  # remove background label
                _pred_logits = pred_logits[roi_idx, _target_labels]
            else:
                _pred_logits = pred_logits.squeeze(dim=1)

            mask_losses = {
                f"mask{self.loss_name}": self.loss(_pred_logits, target_masks, **kwargs),
            }
        else:
            mask_losses = {f"mask{self.loss_name}": pred_logits.new_zeros([1])}
        return mask_losses

    def logits_to_probs(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        Convert mask logits to probabilities

        Args:
            logits: mask logits [N, C, dims], N=number of RoIs,
                C=number of classes, dims = spatial dimensions
            labels: predicted label for each mask [N, dims],
                N=number of RoIs, dims are spatial dimensions

        Returns:
            Tensor: predicted mask probabilities [N, dims], N=number of
                RoIs, dims = spatial dimensions
        """
        if self.is_class_agnostic():
            return self.logits_convert_fn(logits).squeeze(dim=1)
        else:
            roi_idx = torch.arange(logits.shape[0])
            return self.logits_convert_fn(logits[roi_idx, labels])


class BCEAgnosticMasker(Masker):
    def __init__(
        self,
        conv: CONVGEN,
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        num_convs: int = 3,
        add_norm: bool = False,
        prior_prob: Optional[float] = None,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        reduction: str = "mean",
        **kwargs,
    ):
        """
        Mask head of RCNN
        c_in -> c_internal x num_convs -> conv_transpose -> c_out

        Trained with Binary Cross Entropy Loss. Outputs contain a single
        channel for each RoI.

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of internal channels
            num_classes: number of foreground classes
            num_convs: Number of internal convolutions. Defaults to 3.
            add_norm: Add normalisation layers to internal convolutions.
                Defaults to False.
            prior_prob: prior probability to initalize final convolution
                same as Focal loss init. If None, no init will be performed.
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            reduction: reduction of loss. Refer to
                `nndet.losses.ops.reduction_helper` for all available options.
            kwargs: passed as keyword argument to in-/internal convolutions
        """
        self.prior_prob = prior_prob
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            internal_channels=internal_channels,
            num_classes=num_classes,
            num_convs=num_convs,
            add_norm=add_norm,
            **kwargs,
        )
        self.loss = BCEMaskLoss(
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
            reduction=reduction,
        )
        self.loss_name = "_bce"
        self.logits_convert_fn = torch.nn.Sigmoid()

    @classmethod
    def is_class_agnostic(cls) -> bool:
        """
        Indicate if mask head is class agnositic or class specific

        Returns:
            bool: `True` if predictions will be class agnostic, meaning that
                each RoI produces a mask with a single channel. If `False`
                each RoI produces a mask for each possible class. The loss
                is only computed on the matched label.
        """
        return True

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


class BDiCEAgnosticMasker(BCEAgnosticMasker):
    def __init__(
        self,
        conv: CONVGEN,
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        num_convs: int = 3,
        add_norm: bool = False,
        prior_prob: Optional[float] = None,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        dict_loss_weight: float = 1.0,
        batch_dice: bool = False,
        smooth_nom: float = 0,
        smooth_denom: float = 1e-6,
        reduction: str = "mean",
        **kwargs,
    ):
        """
        Mask head of RCNN
        c_in -> c_internal x num_convs -> conv_transpose -> c_out

        Trained with Binary Cross Entropy Loss and Dice Loss.
        Outputs contain a single channel for each RoI.

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of internal channels
            num_classes: number of foreground classes
            num_convs: Number of internal convolutions. Defaults to 3.
            add_norm: Add normalisation layers to internal convolutions.
                Defaults to False.
            prior_prob: prior probability to initalize final convolution
                same as Focal loss init. If None, no init will be performed.
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            batch_dice: Compute dice statistics across the entire batch
            smooth_nom: numerical constant added to nominator of dice loss
            smooth_denom: numerical constant added to denominator of dice loss
            reduction: reduction of loss. Refer to
                `nndet.losses.ops.reduction_helper` for all available options.
            kwargs: passed as keyword argument to in-/internal convolutions
        """
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            internal_channels=internal_channels,
            num_classes=num_classes,
            num_convs=num_convs,
            add_norm=add_norm,
            prior_prob=prior_prob,
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
            reduction=reduction,
            **kwargs,
        )
        self.loss_dice = BDiceMaskLoss(
            batch_dice=batch_dice,
            smooth_nom=smooth_nom,
            smooth_denom=smooth_denom,
            loss_fp32=loss_fp32,
            loss_weight=dict_loss_weight,
            reduction=reduction,
        )

    def compute_loss(
        self,
        pred_logits: Tensor,
        target_masks: Tensor,
        target_labels: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Base classifier with cross entropy loss (in general hard negative
        example mining should be done before this)

        Args:
            pred_logits: predicted mask logits [N, num_classes, dims],
                where N is the number of RoIs, num_classes is the number of
                foreground classes and dims are spatial dimensions
            target_masks: mask targets as binary masks [N, dims]
                where N is the number of RoIs and dims are spatial dimensions
            target_labels: classification label for each mask [N]
                where N is the number of RoIs (0 is background)

        Returns:
            Tensor: loss
        """
        if pred_logits.numel() > 0:
            if not self.is_class_agnostic():
                roi_idx = torch.arange(pred_logits.shape[0])
                _target_labels = target_labels - 1  # remove background label
                _pred_logits = pred_logits[roi_idx, _target_labels]
            else:
                _pred_logits = pred_logits.squeeze(dim=1)

            mask_losses = {
                "mask_bce": self.loss(_pred_logits, target_masks, **kwargs),
                "mask_dice": self.loss_dice(_pred_logits, target_masks, **kwargs),
            }
        else:
            mask_losses = {
                "mask_bce": pred_logits.new_zeros([1]),
                "mask_dice": pred_logits.new_zeros([1]),
            }
        return mask_losses


class BCESpecificMasker(BCEAgnosticMasker):
    @classmethod
    def is_class_agnostic(cls) -> bool:
        """
        Indicate if mask head is class agnositic or class specific

        Returns:
            bool: `True` if predictions will be class agnostic, meaning that
                each RoI produces a mask with a single channel. If `False`
                each RoI produces a mask for each possible class. The loss
                is only computed on the matched label.
        """
        return False


class BDiCESpecificMasker(BCEAgnosticMasker):
    @classmethod
    def is_class_agnostic(cls) -> bool:
        """
        Indicate if mask head is class agnositic or class specific

        Returns:
            bool: `True` if predictions will be class agnostic, meaning that
                each RoI produces a mask with a single channel. If `False`
                each RoI produces a mask for each possible class. The loss
                is only computed on the matched label.
        """
        return False
