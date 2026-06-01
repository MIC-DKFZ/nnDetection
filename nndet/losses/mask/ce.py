# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
from torch.cuda.amp import autocast

from nndet.losses.ops import Loss, reduction_helper


class BCEMaskLoss(Loss):
    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        reduction: str = "mean",
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        """
        BCE Loss wrapper from PyTorch for binary inputs

        Args:
            pos_weight: equivalent to pos_weight parameter of BCE loss of
                pytorch (weights positive class)
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            reduction: reduction of loss. Refer to
                `nndet.losses.ops.reduction_helper` for all available options.
            weight: equivalent to weight parameter of BCE loss of pytorch
                (weights batch elements)
        """
        super().__init__(
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
            reduction=reduction,
        )
        self.register_buffer("pos_weight", pos_weight)
        self.pos_weight: Optional[torch.Tensor]
        self.register_buffer("weight", weight)
        self.weight: Optional[torch.Tensor]
        if pos_weight is not None:
            raise NotImplementedError("Not implemented. PyTorch interprets last channels as classes.")

    @property
    def reduction(self) -> str:
        if self.torch_reduction.lower() == "mean":
            return self.torch_reduction
        else:
            return self.helper_reduction

    @reduction.setter
    def reduction(self, key: str):
        _key = key.lower()
        if _key == "mean":
            self.torch_reduction = "mean"
            self.helper_reduction = "none"
        else:
            self.torch_reduction = "none"
            self.helper_reduction = _key

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Loss

        Args:
            preds: predictions (pre act). [N, C, *], where N is the batch
                size, C is the number of classes, * are arbitrary spatial
                dimensions
            targets: targets encoded as one hot. [N, C, *], where
                N is the batch size, C is the number of classes, * are
                arbitrary spatial dimensions

        Returns:
            torch.Tensor: computed loss
        """
        _fn = torch.nn.functional.binary_cross_entropy_with_logits
        if self.loss_fp32:
            with autocast(enabled=False):
                loss = _fn(
                    preds.float(),
                    targets.float(),
                    weight=self.weight,
                    pos_weight=self.pos_weight,
                    reduction=self.torch_reduction,
                )
        else:
            loss = _fn(
                preds,
                targets,
                weight=self.weight,
                pos_weight=self.pos_weight,
                reduction=self.torch_reduction,
            )
        return self.loss_weight * reduction_helper(loss, reduction=self.helper_reduction)

    def extra_repr(self) -> str:
        return (
            f"weight={self.weight}, "
            f"loss_weight={self.loss_weight}, "
            f"loss_fp32={self.loss_fp32}, "
            f"reduction={self.reduction}"
        )
