import torch

from nndet.core.boxes.ops import generalized_box_iou_3d_paired
from nndet.losses.ops import Loss, reduction_helper


class GIoULoss(Loss):
    def __init__(
        self,
        eps: float = 1e-7,
        loss_weight: float = 1.0,
        loss_fp32: bool = True,
        reduction: str = "none",
    ):
        """
        Generalized IoU Loss
        `Generalized Intersection over Union: A Metric and A Loss for Bounding
        Box Regression` https://arxiv.org/abs/1902.09630

        Args:
            eps: small constant for numerical stability
            loss_weight: scalar to balance multiple losses
            loss_fp32: IGNORED, loss is always computed in fp32. This argument
                is only added here to have a uniform API.

        Notes:
            Original paper uses lambda=10 to balance regression and cls losses
            for PASCAL VOC and COCO (not tuned for coco)

            `End-to-End Object Detection with Transformers` https://arxiv.org/abs/2005.12872
            "Our enhanced Faster-RCNN+ baselines use GIoU [38] loss along with
            the standard l1 loss for bounding box regression. We performed a grid search
            to find the best weights for the losses and the final models use only GIoU loss
            with weights 20 and 1 for box and proposal regression tasks respectively"
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
        loss = reduction_helper(
            generalized_box_iou_3d_paired(
                boxes1=pred_boxes,
                boxes2=target_boxes,
                eps=self.eps,
            ),
            reduction=self.reduction,
        )
        return self.loss_weight * -1 * loss

        # torch.diag(
        #     generalized_box_iou(pred_boxes, target_boxes, eps=self.eps), diagonal=0
        # ),
