import torch
from loguru import logger
from torch.cuda.amp import autocast

from nndet.losses.ops import one_hot_smooth_first


class CESegLoss(torch.nn.CrossEntropyLoss):
    def __init__(
        self,
        *args,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ) -> None:
        """
        Same as CE from pytorch
        Targets can be float or long, it is castet to the correct type

        Args:
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
        """
        super().__init__(
            *args,
            **kwargs,
        )
        self.loss_weight = loss_weight
        self.loss_fp32 = loss_fp32
        if loss_fp32:
            logger.info(f"{self.__class__.__name__} uses FP32 loss computation.")

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Same as CE from pytorch
        """
        if self.loss_fp32:
            with autocast(enabled=False):
                loss = self.loss_weight * super().forward(input.float(), target.long())
        else:
            loss = self.loss_weight * super().forward(input, target.long())
        return loss


class BCESegLoss(torch.nn.BCEWithLogitsLoss):
    def __init__(
        self,
        *args,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ) -> None:
        """
        Same as BCE with Logits from pytorch
        Targets can be float or long, it is castet to the correct type

        Args:
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
        """
        super().__init__(
            *args,
            **kwargs,
        )
        self.loss_weight = loss_weight
        self.loss_fp32 = loss_fp32
        if loss_fp32:
            logger.info(f"{self.__class__.__name__} uses FP32 loss computation.")

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Same as BCE with Logits from pytorch
        """
        num_classes = input.shape[1]
        _target = one_hot_smooth_first(target, num_classes=num_classes + 1)
        _target = _target[:, 1:]

        if self.loss_fp32:
            with autocast(enabled=False):
                loss = self.loss_weight * super().forward(input.float(), _target.float())
        else:
            loss = self.loss_weight * super().forward(input, _target.float())
        return loss
