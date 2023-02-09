from loguru import logger
from torch import Tensor

from nndet.losses.ops import one_hot_smooth_first
from nndet.losses.segmentation.ce import BCESegLoss, CESegLoss


class TopKCESegLoss(CESegLoss):
    def __init__(
        self,
        topk: float,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ):
        """
        Uses topk percent of values to compute CE loss
        (expects pre softmax logits!)

        Args:
            topk: percentage of all entries to use for loss computation
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
        """
        if "reduction" in kwargs:
            raise ValueError("Reduction is not supported in TopKLoss." "This will always return the mean!")
        super().__init__(
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
            reduction="none",
            **kwargs,
        )
        if topk < 0 or topk > 1:
            raise ValueError("topk needs to be in the range [0, 1].")
        self.topk = topk
        logger.info(f"TopK loss uses topk: {self.topk:.2f}")

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Compute CE loss and uses mean of topk percent of the entries

        Args:
            input: logits for all foreground classes [N, C, * ]
            target: target classes. 0 is treated as background, >0 are
                treated as foreground classes. [N, * ]

        Returns:
            Tensor: final loss
        """
        losses = super().forward(input, target)

        k = int(max(losses.numel() * self.topk, 1))
        return losses.view(-1).topk(k=k, sorted=False)[0].mean()


class TopKBCESegLoss(BCESegLoss):
    def __init__(
        self,
        num_classes: int,
        topk: float,
        smoothing: float = 0.0,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ):
        """
        Uses topk percent of values to compute BCE loss with one hot
        (support multi class through one hot, expects pre sigmoid logits!)

        Args:
            num_classes: number of classes
            topk: percentage of all entries to use for loss computation
            smoothing:  label smoothing
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
        """
        if "reduction" in kwargs:
            raise ValueError("Reduction is not supported in TopKLoss." "This will always return the mean!")
        super().__init__(
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
            reduction="none",
            **kwargs,
        )
        self.smoothing = smoothing
        if smoothing > 0:
            logger.info(f"Running label smoothing with smoothing: {smoothing}")
        self.num_classes = num_classes

        self.topk = topk

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Compute BCE loss based on one hot encoding of foreground(!) classes
        and uses mean of topk percent of the entries

        Args:
            input: logits for all foreground(!) classes [N, C, * ]
            target: target classes [N, * ]. Targets will be encoded with one
                hot and 0 is treated as the background class and removed.

        Returns:
            Tensor: final loss
        """
        target_one_hot = one_hot_smooth_first(
            target, num_classes=self.num_classes + 1, smoothing=self.smoothing
        )  # [N, C + 1]
        target_one_hot = target_one_hot[:, 1:]  # background is implicitly encoded
        losses = super().forward(input, target_one_hot.float())

        k = int(max(losses.numel() * self.topk, 1))
        return losses.view(-1).topk(k=k, sorted=False)[0].mean()
