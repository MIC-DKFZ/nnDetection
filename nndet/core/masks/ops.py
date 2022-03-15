from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from nndet.core.boxes.ops import box_size


def roi_mask_to_image_mask(
    boxes: torch.Tensor,
    masks: torch.Tensor,
    image_shape: Tuple[Tuple[int, int], Tuple[int, int, int]],
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    antialias: bool = False,
    threshold: Optional[float] = None,
) -> torch.Tensor:
    # TODO: unit test with empty mask, check shape == 0
    # TODO: boxes rounding
    # TODO: check for float
    assert boxes.shape[0] == masks.shape[0]
    num_items = boxes.shape[0] if boxes.numel() > 0 else 0

    image_mask = torch.zeros(num_items, *image_shape, device=masks.device)
    if num_items == 0:
        return image_mask

    boxes_size = torch.round(box_size(boxes)).to(dtype=torch.int)
    for idx in range(num_items):
        _mask_rescale = F.interpolate(
            masks[idx][None],
            size=tuple(boxes_size[idx].tolist()),
            mode=mode,
            align_corners=align_corners,
            # antialias=antialias,
        )
        image_coords = [
            slice(int(boxes[idx, 0]), int(boxes[idx, 0]) + int(boxes_size[idx, 0])),
            slice(int(boxes[idx, 1]), int(boxes[idx, 1]) + int(boxes_size[idx, 1])),
        ]
        if boxes.shape[1] == 6:
            image_coords.append(
                slice(int(boxes[idx, 4]), int(boxes[idx, 4]) + int(boxes_size[idx, 2]))
            )
        image_mask[idx][tuple(image_coords)] = _mask_rescale[0, 0]

    if threshold is not None:
        image_mask = (image_mask > threshold).to(dtype=torch.float)
    return image_mask


def bin_mask_iou(
    bin_masks1: torch.Tensor,
    bin_masks2: torch.Tensor,
) -> torch.Tensor:
    bin_masks1_flattened = bin_masks1.flatten(1)
    bin_masks2_flattened = bin_masks2.flatten(1)

    masks1_vol = bin_masks1_flattened.sum(dim=1)  # [N]
    masks2_vol = bin_masks2_flattened.sum(dim=1)  # [M]

    intersection = torch.mm(bin_masks1_flattened, bin_masks2_flattened.T)  # [N, M]
    union = masks1_vol[:, None] + masks2_vol[None] - intersection  # [N, M]
    return intersection / union
