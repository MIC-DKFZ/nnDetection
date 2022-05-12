import torch

from nndet.losses.ops import Loss
from nndet.losses.regression.functional.diou import distance_iou_loss_3d


class DIoULoss(Loss):
    def __init__(
        self,
        eps: float = 1e-7,
        loss_weight: float = 1.0,
        loss_fp32: bool = True,
        reduction: str = "none",
    ):
        """
        Distance IoU Loss
        `Distance-IoU Loss: Faster and Better Learning for Bounding Box
        Regression` https://arxiv.org/abs/1911.08287

        Args:
            eps: small constant for numerical stability
            loss_weight: scalar to balance multiple losses
            loss_fp32: IGNORED, loss is always computed in fp32. This argument
                is only added here to have a uniform API.
            reduction: 'mean'|'sum'|'none'
                mean: mean of loss over entire batch
                sum: sum of loss over entire batch
                none: no reduction

        Notes:
            Weight was set to 5 in paper (observed improved perf with higher
            weight in dense detectors)

        """
        super().__init__(
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
            reduction=reduction,
        )
        self.eps = eps

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute generalized iou loss

        Args:
            pred_boxes: predicted boxes (x1, y1, x2, y2, (z1, z2)) [N, dim * 2]
            target_boxes: target boxes (x1, y1, x2, y2, (z1, z2)) [N, dim * 2]

        Returns:
            Tensor: loss
        """
        return self.loss_weight * distance_iou_loss_3d(
            pred_boxes=pred_boxes,
            target_boxes=target_boxes,
            eps=self.eps,
        )
