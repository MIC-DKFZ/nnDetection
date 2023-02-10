import torch

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
        TopK with CE Loss

        Args:
            topk: percentage of all entries to use for loss computation
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            kwargs: passed to `nndet.losses.segmentation.ce.CESegLoss`
        """
        if "reduction" in kwargs and not kwargs.pop("reduction") == "mean":
            raise ValueError("TopK Loss only supports 'mean' reduction")
        super().__init__(
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
            reduction="none",
            **kwargs,
        )
        if topk < 0 or topk > 1:
            raise ValueError("topk needs to be in the range [0, 1].")
        self.topk = topk

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Loss

        Args:
            preds: predictions (without act). [N, C, *], where N is the batch
                size, C is the number of classes, * are arbitrary spatial
                dimensions
            targets: numerical target values. [N, *], where N is the batch
                size, * are arbitrary spatial dimensions

        Returns:
            torch.Tensor: computed loss
        """
        losses = super().forward(preds, targets)
        k = int(max(losses.numel() * self.topk, 1))
        return losses.view(-1).topk(k=k, sorted=False)[0].mean()

    def extra_repr(self) -> str:
        return f"topk={self.topk}"


class TopKBCESegLoss(BCESegLoss):
    def __init__(
        self,
        topk: float,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ):
        """
        Uses topk percent of values to compute BCE loss with one hot
        (support multi class through one hot, expects pre sigmoid logits!)

        Args:
            topk: percentage of all entries to use for loss computation
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            kwargs: passed to `nndet.losses.segmentation.ce.BCESegLoss`
        """
        if "reduction" in kwargs and not kwargs.pop("reduction") == "mean":
            raise ValueError("TopK Loss only supports 'mean' reduction")
        super().__init__(
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
            reduction="none",
            **kwargs,
        )
        self.topk = topk

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Loss

        Args:
            preds: predictions (without act). [N, C, *], where N is the batch
                size, C is the number of classes, * are arbitrary spatial
                dimensions
            targets: numerical target values. [N, *], where N is the batch
                size, * are arbitrary spatial dimensions

        Returns:
            torch.Tensor: computed loss
        """
        losses = super().forward(preds, targets)
        k = int(max(losses.numel() * self.topk, 1))
        return losses.view(-1).topk(k=k, sorted=False)[0].mean()

    def extra_repr(self) -> str:
        return f"topk={self.topk}"
