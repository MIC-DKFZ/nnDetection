import torch
from loguru import logger
from torch.cuda.amp import autocast


class BCEMaskLoss(torch.nn.BCEWithLogitsLoss):
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
        if self.loss_fp32:
            with autocast(enabled=False):
                loss = self.loss_weight * super().forward(input.float(), target.float())
        else:
            loss = self.loss_weight * super().forward(input, target)
        return loss
