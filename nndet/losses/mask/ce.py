from typing import Optional

import torch
from torch.cuda.amp import autocast

from nndet.losses.ops import Loss, reduction_helper


class BCEMaskLoss(Loss):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        reduction: str = "mean",
    ) -> None:
        """
        BCE Loss wrapper from PyTorch for binary inputs

        Args:
            weight: weiught for BCE loss, see PyTorch docs for more info.
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            reduction: reduction of loss. Refer to
                `nndet.losses.ops.reduction_helper` for all available options.
        """
        super().__init__(
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
            reduction=reduction,
        )
        self.weight = weight

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
                    reduction="none",
                )
        else:
            loss = _fn(
                preds,
                targets,
                weight=self.weight,
                reduction="none",
            )
        return self.loss_weight * reduction_helper(loss, reduction=self.reduction)

    def extra_repr(self) -> str:
        return (
            f"weight={self.weight}"
            f"loss_weight={self.loss_weight}, "
            f"loss_fp32={self.loss_fp32}, "
            f"reduction={self.reduction}"
        )
