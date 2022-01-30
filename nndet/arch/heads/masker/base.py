from abc import abstractmethod
from typing import Optional, TypeVar

import torch
from torch import Tensor, nn

from nndet.arch.heads.abstract import Classifier
from nndet.losses.classification import BCEWithLogitsLoss
from nndet.losses.segmentation import SoftDiceLoss

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
        # self.init_weights()

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
                kernel_size=3,
                stride=1,
                padding=1,
                add_norm=False,
                add_act=False,
                bias=True,
            ),
        )
        return _conv_out

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

    @classmethod
    def class_agnostic(cls):
        return True


class BCESingleMasker(Masker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = BCEWithLogitsLoss()
        self.logits_convert_fn = torch.nn.Sigmoid()

    def get_output_channels(self) -> int:
        return 1


class DiceBCESingleMasker(BCESingleMasker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: cleanup
        self.loss_dice = SoftDiceLoss(
            nonlin=self.logits_convert_fn,
            batch_dice=False,
            do_bg=True,
            smooth_nom=1e-5,
            smooth_denom=1e-5,
            loss_weight=1.0,
            loss_fp32=True,
            reduction="mean",
        )

    def get_output_channels(self) -> int:
        return 1

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
