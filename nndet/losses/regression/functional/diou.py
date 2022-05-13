import torch
from torch.cuda.amp import autocast

from nndet.core.boxes.ops import distance_box_iou_3d_paired
from nndet.losses.ops import reduction_helper


@autocast(enabled=False)
def distance_iou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    reduction: str,
    eps: float = 0,
) -> torch.Tensor:
    """
    Distance IoU Loss
    L = 1 - IoU + d^2(c, c_gt) / diag_enclosing^2

    Args:
        pred_boxes: predicted boxes [N, dims] (x1, y1, x2, y2, z1, z2)
        target_boxes: target boxes [N, dims] (x1, y1, x2, y2, z1, z2)
        eps: small constant for numerical stability
        reduction: 'mean'|'sum'|'none'
            mean: mean of loss over entire batch
            sum: sum of loss over entire batch
            none: no reduction

    Returns:
        torch.Tensor: computed loss

    Notes:
        Need to compute IoU in float32 (autocast=False) because the
        volume/area can be to large
    """
    if pred_boxes.nelement() == 0 or target_boxes.nelement() == 0:
        return torch.tensor([]).to(pred_boxes)
    if pred_boxes.shape[-1] == 4:
        raise NotImplementedError("DIoU Loss not implemented for 2D")
    else:
        loss = distance_box_iou_3d_paired(
            boxes1=pred_boxes.float(),
            boxes2=target_boxes.float(),
            eps=eps,
        )
    return reduction_helper(loss, reduction=reduction)
