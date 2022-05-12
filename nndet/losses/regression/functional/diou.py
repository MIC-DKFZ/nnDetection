import torch

from nndet.core.boxes.ops import box_center, box_iou_union_3d_paired
from nndet.losses.ops import reduction_helper


def distance_iou_loss_3d(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    reduction: str,
    eps: float = 0.0,
) -> torch.Tensor:
    """
    Distance IoU Loss
    L = 1 - IoU + d^2(c, c_gt) / diag_enclosing^2

    Args:
        pred_boxes: predicted boxes [6, dims] (x1, y1, x2, y2, z1, z2)
        target_boxes: target boxes [6, dims] (x1, y1, x2, y2, z1, z2)

    Returns:
        torch.Tensor: computed loss
    """
    iou, _ = box_iou_union_3d_paired(pred_boxes, target_boxes, eps=eps)  # [N]
    dc = (box_center(pred_boxes) - box_center(target_boxes)).pow(2).sum(dim=1)  # [N]

    # enclosing box
    x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])  # [N]
    y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])  # [N]
    x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])  # [N]
    y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])  # [N]
    z1 = torch.min(pred_boxes[:, 4], target_boxes[:, 4])  # [N]
    z2 = torch.max(pred_boxes[:, 5], target_boxes[:, 5])  # [N]
    diag = (
        (x2 - x1).clamp(min=0).pow(2)
        + (y2 - y1).clamp(min=0).pow(2)
        + (z2 - z1).clamp(min=0).pow(2)
        + eps
    )
    loss = 1 - iou + dc / diag
    return reduction_helper(loss, reduction=reduction)
