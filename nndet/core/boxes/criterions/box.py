import torch

import nndet.core.ops_torch as ops_torch
from nndet.core.boxes.criterions.base import BoxCriterion


class L1RegCriterion(BoxCriterion):
    def __init__(self, loss_weight: float) -> None:
        """
        Compute L1 based box cost matrix

        Args:
            loss_weight: weighting for computed loss
        """
        super().__init__(loss_weight=loss_weight)

    def forward(
        self,
        pred_coords: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute l1 box criterion

        Args:
            pred_coords: predicted bounding box coords [B * R, dims * 2]
                where B=batch size, R=number of predictions, dims=number of
                spatial dimensions (format corresponds to model format)
            target_labels: target ground truth boxes [L, dims * 2] where
                L is the number of ground truth objects  (format corresponds
                to model format)

        Returns:
            torch.Tensor: cost matrix [B * R, L], where B=batch size,
                R=number of predictions, L is the number of ground truth
                objects
        """
        return self.loss_weight * torch.cdist(pred_coords, target_boxes, p=1)


class GIoUCenterBoxCriterion(BoxCriterion):
    def __init__(self, loss_weight: float, eps: float = 1e-6) -> None:
        """
        Compute L1 based box cost matrix

        Args:
            loss_weight: weighting for computed loss
            eps: passed to GIoU computation for numerical stability
        """
        super().__init__(loss_weight=loss_weight)
        self.eps = eps

    def forward(
        self,
        pred_coords: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute l1 box criterion

        Args:
            pred_coords: predicted bounding box coords [B * R, dims * 2]
                where B=batch size, R=number of predictions, dims=number of
                spatial dimensions (format corresponds to model format)
            target_labels: target ground truth boxes [L, dims * 2] where
                L is the number of ground truth objects  (format corresponds
                to model format)

        Returns:
            torch.Tensor: cost matrix [B * R, L], where B=batch size,
                R=number of predictions, L is the number of ground truth
                objects
        """
        return (
            self.loss_weight
            * -1
            * ops_torch.generalized_box_iou(
                ops_torch.box_center2point_format(pred_coords),
                ops_torch.box_center2point_format(target_boxes),
                eps=self.eps,
            )
        )
