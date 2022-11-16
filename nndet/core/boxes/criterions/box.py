import torch

from nndet.core.boxes.criterions.base import BoxCriterion


class L1BoxCriterion(BoxCriterion):
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
