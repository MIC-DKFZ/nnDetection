# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import List, Sequence

import numpy as np
import torch
from scipy.stats import norm
from torch import Tensor


def apply_offsets_to_boxes(
    boxes: List[Tensor],
    tile_offset: Sequence[Sequence[int]],
) -> List[Tensor]:
    """
    Apply offset to bounding boxes to position them correctly inside
    the whole case

    Args:
        boxes: predicted boxes [N, dims * 2]
            [x1, y1, x2, y2, (z1, z2))
        tile_offset: defines offset for each tile

    Returns:
        List[Tensor]: bounding boxes with respect to origin of whole case
    """
    offset_boxes = []
    for img_boxes, offset in zip(boxes, tile_offset):
        if img_boxes.nelement() == 0:
            offset_boxes.append(img_boxes)
            continue
        offset = Tensor(offset).to(img_boxes)
        _boxes = img_boxes.clone()

        _boxes[:, 0] += offset[0]
        _boxes[:, 1] += offset[1]
        _boxes[:, 2] += offset[0]
        _boxes[:, 3] += offset[1]

        if img_boxes.shape[1] == 6:
            _boxes[:, 4] += offset[2]
            _boxes[:, 5] += offset[2]

        offset_boxes.append(_boxes)
    return offset_boxes


def get_box_in_tile_weight_linear(
    box_centers: Tensor,
    tile_size: Sequence[int],
    plateau_length: float,
) -> Tensor:
    """
    Assign boxes near the corner a lower weight.
    The midle has a plateau with weight one, starting from patchsize / 2
    the weights decreases linearly until 0.5 is reached.

    Args:
        box_centers: center predicted box [N, dims]
        tile_size: size the of patch/tile
        plateau_length: adjust width of plateau and min weight

    Returns:
        Tensor: weight for each bounding box [N]
    """
    if box_centers.numel() > 0:
        tile_center = torch.tensor(tile_size).to(box_centers) / 2.0  # [dims]

        max_dist = tile_center.norm(p=2)  # [1]
        boxes_dist = (box_centers - tile_center[None]).norm(p=2, dim=1)  # [N]
        weight = -(boxes_dist / max_dist - plateau_length).clamp_(min=0) + 1
        return weight
    else:
        return Tensor([]).to(box_centers)


def get_box_in_tile_weight_normal(
    box_centers: Tensor,
    tile_size: Sequence[int],
) -> Tensor:
    """
    Assign boxes at the corners of tiles a lower weight (weight
    is drawn form a scaled normal distribution)

    Args:
        box_centers: center predicted box [N, dims]
        tile_size: size the of patch/tile

    Returns:
        Tensor: weight for each bounding box [N]
    """
    if box_centers.numel() > 0:
        all_weights = []
        centers_np = box_centers.detach().cpu().numpy()
        for center_np in centers_np:
            weight = np.mean(
                [
                    norm.pdf(bc, loc=ps, scale=ps * 0.8) * np.sqrt(2 * np.pi) * ps * 0.8
                    for bc, ps in zip(center_np, np.array(tile_size) / 2)
                ]
            )
            all_weights.append([weight])
        return torch.from_numpy(np.concatenate(all_weights)).to(box_centers)
    else:
        return Tensor([]).to(box_centers)
