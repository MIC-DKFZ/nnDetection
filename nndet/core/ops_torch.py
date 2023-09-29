# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

# ivnerse_sigmoid function from
# https://github.com/fundamentalvision/Deformable-DETR/blob/11169a60c33333af00a4849f1808023eba96a931/util/misc.py  # noqa: E501
# SPDX-FileCopyrightText: 2020 SenseTime
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from numpy import ndarray
from torch import Tensor
from torch.cuda.amp import autocast

from nndet.utils.tensor import ensure_min_float32
from nndet.utils.typing import ND_TUPLE_INT


@autocast(enabled=False)
def box_area(
    boxes: Tensor,
) -> Tensor:
    """
    Computes the area of a set of bounding boxes

    Args:
        boxes: boxes of shape; (x1, y1, x2, y2, (z1, z2))[N, dim * 2]

    Returns:
        Tensor: area of boxes

    See Also:
        :func:`box_area_3d`, :func:`torchvision.ops.boxes.box_area`
    """
    _boxes = ensure_min_float32(boxes)
    if boxes.shape[-1] == 4:
        return box_area_2d(_boxes)
    else:
        return box_area_3d(_boxes)


@autocast(enabled=False)
def box_iou(
    boxes1: Tensor,
    boxes2: Tensor,
    eps: float = 0,
) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes1: boxes (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        boxes2: boxes (x1, y1, x2, y2, (z1, z2))[M, dim * 2]
        eps: optional small constant for numerical stability

    Returns:
        Tensor: the NxM iou matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2; [N, M]

    See Also:
        :func:`box_iou_3d`, :func:`torchvision.ops.boxes.box_iou`

    Notes:
        Need to compute IoU in float32 (autocast=False) because the
        volume/area can be to large
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.tensor([]).to(boxes1)

    _boxes1 = ensure_min_float32(boxes1)
    _boxes2 = ensure_min_float32(boxes2)

    if boxes1.shape[-1] == 4:
        return box_iou_union_2d(_boxes1, _boxes2, eps=eps)[0]
    else:
        return box_iou_union_3d(_boxes1, _boxes2, eps=eps)[0]


@autocast(enabled=False)
def generalized_box_iou(
    boxes1: Tensor,
    boxes2: Tensor,
    eps: float = 0,
) -> Tensor:
    """
    Generalized box iou

    Args:
        boxes1: boxes (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        boxes2: boxes (x1, y1, x2, y2, (z1, z2))[M, dim * 2]
        eps: optional small constant for numerical stability

    Returns:
        Tensor: the NxM iou matrix containing the pairwise
            generalized IoU values for every element in boxes1 and boxes2; [N, M]

    Notes:
        Need to compute IoU in float32 (autocast=False) because the
        volume/area can be to large
    """
    if boxes1.nelement() == 0 or boxes2.nelement() == 0:
        return torch.tensor([]).to(boxes1)

    _boxes1 = ensure_min_float32(boxes1)
    _boxes2 = ensure_min_float32(boxes2)

    if boxes1.shape[-1] == 4:
        return generalized_box_iou_2d(_boxes1, _boxes2, eps=eps)
    else:
        return generalized_box_iou_3d(_boxes1, _boxes2, eps=eps)


@autocast(enabled=False)
def box_iou_paired(
    boxes1: Tensor,
    boxes2: Tensor,
    eps: float = 0,
) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) and Union of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2, z1, z2) format.

    Args:
        boxes1: set of boxes (x1, y1, x2, y2, z1, z2)[N, 6]
        boxes2: set of boxes (x1, y1, x2, y2, z1, z2)[N, 6]
        eps: optional small constant for numerical stability

    Returns:
        Tensor: vector [N] containing the boxes between box sets
        Tensor: vector [N] containing the union between the box sets

    Notes:
        Need to compute IoU in float32 (autocast=False) because the
        volume/area can be to large
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.tensor([]).to(boxes1)

    _boxes1 = ensure_min_float32(boxes1)
    _boxes2 = ensure_min_float32(boxes2)

    if boxes1.shape[-1] == 4:
        raise NotImplementedError("2D case not implemented")
    else:
        return box_iou_union_3d_paired(_boxes1, _boxes2, eps=eps)[0]


@autocast(enabled=False)
def generalized_box_iou_paired(
    boxes1: Tensor,
    boxes2: Tensor,
    eps: float = 0,
) -> Tensor:
    """
    Computes the generalized box iou between given bounding boxes
    in a paired fashion

    Args:
        boxes1: set of boxes (x1, y1, x2, y2, z1, z2)[N, 6]
        boxes2: set of boxes (x1, y1, x2, y2, z1, z2)[N, 6]
        eps: optional small constant for numerical stability

    Returns:
        Tensor: vector [N] containing the pairwise generalized IoU values
            for every element in boxes1 and boxes2

    Notes:
        Need to compute IoU in float32 (autocast=False) because the
        volume/area can be to large
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.tensor([]).to(boxes1)

    _boxes1 = ensure_min_float32(boxes1)
    _boxes2 = ensure_min_float32(boxes2)

    if boxes1.shape[-1] == 4:
        raise NotImplementedError("2D case not implemented")
    else:
        return generalized_box_iou_3d_paired(_boxes1, _boxes2, eps=eps)


@autocast(enabled=False)
def distance_box_iou_paired(
    boxes1: Tensor,
    boxes2: Tensor,
    eps: float = 0,
) -> Tensor:
    """
    Distance IoU Loss
    L = 1 - IoU + d^2(c, c_gt) / diag_enclosing^2

    Args:
        boxes1: predicted boxes [N, dims] (x1, y1, x2, y2, z1, z2)
        boxes2: target boxes [N, dims] (x1, y1, x2, y2, z1, z2)
        eps: small constant for numerical stability

    Returns:
        torch.Tensor: computed loss [N]

    Notes:
        Need to compute IoU in float32 (autocast=False) because the
        volume/area can be to large
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.tensor([]).to(boxes1)

    _boxes1 = ensure_min_float32(boxes1)
    _boxes2 = ensure_min_float32(boxes2)

    if boxes1.shape[-1] == 4:
        raise NotImplementedError("2D case not implemented")
    else:
        return distance_box_iou_3d_paired(_boxes1, _boxes2, eps=eps)


def box_area_3d(
    boxes: Tensor,
) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2, z1, z2) coordinates.

    Args:
        boxes: boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2, z1, z2) format. [N, 6]

    Returns:
        Tensor: area for each box [N]

    Notes:
        always prefer using the n-D version since it takes care of data types.
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 4])


def box_area_2d(
    boxes: Tensor,
) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes: boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format. [N, 4]

    Returns:
        Tensor: area for each box [N]

    Notes:
        always prefer using the n-D version since it takes care of data types.
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou_union_3d(
    boxes1: Tensor,
    boxes2: Tensor,
    eps: float = 0,
) -> Tuple[Tensor, Tensor]:
    """
    Return intersection-over-union (Jaccard index) and  of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2, z1, z2) format.

    Args:
        boxes1: set of boxes (x1, y1, x2, y2, z1, z2)[N, 6]
        boxes2: set of boxes (x1, y1, x2, y2, z1, z2)[M, 6]
        eps: optional small constant for numerical stability

    Returns:
        Tensor: the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2, shape [N, M]
        Tensor: the nxM matrix containing the pairwise union
            values, shape [N, M]

    Notes:
        always prefer using the n-D version since it takes care of data types.
    """
    vol1 = box_area_3d(boxes1)
    vol2 = box_area_3d(boxes2)

    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # [N, M]
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])  # [N, M]
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])  # [N, M]
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])  # [N, M]
    z1 = torch.max(boxes1[:, None, 4], boxes2[:, 4])  # [N, M]
    z2 = torch.min(boxes1[:, None, 5], boxes2[:, 5])  # [N, M]

    inter = ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0) * (z2 - z1).clamp(min=0)) + eps  # [N, M]
    union = vol1[:, None] + vol2 - inter
    return inter / union, union


def box_iou_union_3d_paired(
    boxes1: Tensor,
    boxes2: Tensor,
    eps: float = 0,
) -> Tuple[Tensor, Tensor]:
    """
    Return intersection-over-union (Jaccard index) and Union of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2, z1, z2) format.

    Args:
        boxes1: set of boxes (x1, y1, x2, y2, z1, z2)[N, 6]
        boxes2: set of boxes (x1, y1, x2, y2, z1, z2)[N, 6]
        eps: optional small constant for numerical stability

    Returns:
        Tensor: vector [N] containing the boxes between box sets
        Tensor: vector [N] containing the union between the box sets

    Notes:
        always prefer using the n-D version since it takes care of data types.
    """
    vol1 = box_area_3d(boxes1)  # [N]
    vol2 = box_area_3d(boxes2)  # [N]

    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])  # [N]
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])  # [N]
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])  # [N]
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])  # [N]
    z1 = torch.max(boxes1[:, 4], boxes2[:, 4])  # [N]
    z2 = torch.min(boxes1[:, 5], boxes2[:, 5])  # [N]

    inter = ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0) * (z2 - z1).clamp(min=0)) + eps  # [N]

    union = vol1 + vol2 - inter  # [N]
    return inter / union, union  # [N]


def generalized_box_iou_3d(
    boxes1: Tensor,
    boxes2: Tensor,
    eps: float = 0,
) -> Tensor:
    """
    Computes the generalized box iou between given bounding boxes

    Args:
        boxes1: set of boxes (x1, y1, x2, y2, z1, z2)[N, 6]
        boxes2: set of boxes (x1, y1, x2, y2, z1, z2)[M, 6]
        eps: optional small constant for numerical stability

    Returns:
        Tensor: the NxM matrix containing the pairwise generalized IoU values
            for every element in boxes1 and boxes2, shape [N, M]

    Notes:
        always prefer using the n-D version since it takes care of data types.
    """
    iou, union = box_iou_union_3d(boxes1, boxes2)

    x1 = torch.min(boxes1[:, None, 0], boxes2[:, 0])  # [N, M]
    y1 = torch.min(boxes1[:, None, 1], boxes2[:, 1])  # [N, M]
    x2 = torch.max(boxes1[:, None, 2], boxes2[:, 2])  # [N, M]
    y2 = torch.max(boxes1[:, None, 3], boxes2[:, 3])  # [N, M]
    z1 = torch.min(boxes1[:, None, 4], boxes2[:, 4])  # [N, M]
    z2 = torch.max(boxes1[:, None, 5], boxes2[:, 5])  # [N, M]

    vol = ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0) * (z2 - z1).clamp(min=0)) + eps  # [N, M]
    return iou - (vol - union) / vol


def generalized_box_iou_3d_paired(
    boxes1: Tensor,
    boxes2: Tensor,
    eps: float = 0,
) -> Tensor:
    """
    Computes the generalized box iou between given bounding boxes
    in a paired fashion

    Args:
        boxes1: set of boxes (x1, y1, x2, y2, z1, z2)[N, 6]
        boxes2: set of boxes (x1, y1, x2, y2, z1, z2)[N, 6]
        eps: optional small constant for numerical stability

    Returns:
        Tensor: vector [N] containing the pairwise generalized IoU values
            for every element in boxes1 and boxes2

    Notes:
        always prefer using the n-D version since it takes care of data types.
    """
    iou, union = box_iou_union_3d_paired(boxes1, boxes2)  # [N], [N]

    x1 = torch.min(boxes1[:, 0], boxes2[:, 0])  # [N]
    y1 = torch.min(boxes1[:, 1], boxes2[:, 1])  # [N]
    x2 = torch.max(boxes1[:, 2], boxes2[:, 2])  # [N]
    y2 = torch.max(boxes1[:, 3], boxes2[:, 3])  # [N]
    z1 = torch.min(boxes1[:, 4], boxes2[:, 4])  # [N]
    z2 = torch.max(boxes1[:, 5], boxes2[:, 5])  # [N]

    vol = ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0) * (z2 - z1).clamp(min=0)) + eps  # [N]
    return iou - (vol - union) / vol


def distance_box_iou_3d_paired(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 0.0,
) -> torch.Tensor:
    """
    Distance IoU Loss
    L = 1 - IoU + d^2(c, c_gt) / diag_enclosing^2

    Args:
        boxes1: predicted boxes [N, dims] (x1, y1, x2, y2, z1, z2)
        boxes2: target boxes [N, dims] (x1, y1, x2, y2, z1, z2)
        eps: small constant for numerical stability

    Returns:
        torch.Tensor: computed loss

    Notes:
        always prefer using the n-D version since it takes care of data types.
    """
    iou, _ = box_iou_union_3d_paired(boxes1, boxes2, eps=eps)  # [N]
    dc = (box_center(boxes1) - box_center(boxes2)).pow(2).sum(dim=1)  # [N]

    # enclosing box
    x1 = torch.min(boxes1[:, 0], boxes2[:, 0])  # [N]
    y1 = torch.min(boxes1[:, 1], boxes2[:, 1])  # [N]
    x2 = torch.max(boxes1[:, 2], boxes2[:, 2])  # [N]
    y2 = torch.max(boxes1[:, 3], boxes2[:, 3])  # [N]
    z1 = torch.min(boxes1[:, 4], boxes2[:, 4])  # [N]
    z2 = torch.max(boxes1[:, 5], boxes2[:, 5])  # [N]
    diag = (x2 - x1).pow(2) + (y2 - y1).pow(2) + (z2 - z1).pow(2) + eps
    return 1 - iou + (dc / diag)


def box_iou_union_2d(
    boxes1: Tensor,
    boxes2: Tensor,
    eps: float = 0,
) -> Tuple[Tensor, Tensor]:
    """
    Return intersection-over-union (Jaccard index) and  of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Args:
        boxes1: set of boxes (x1, y1, x2, y2)[N, 4]
        boxes2: set of boxes (x1, y1, x2, y2)[M, 4]
        eps: optional small constant for numerical stability

    Returns:
        Tensor: iou  NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2, shape [N, M]
        Tensor: union NxM matrix containing the pairwise union
            values, shape [N, M]

    Notes:
        always prefer using the n-D version since it takes care of data types.
    """
    area1 = box_area_2d(boxes1)
    area2 = box_area_2d(boxes2)

    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # [N, M]
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])  # [N, M]
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])  # [N, M]
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])  # [N, M]

    inter = ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)) + eps  # [N, M]
    union = area1[:, None] + area2 - inter
    return inter / union, union


def generalized_box_iou_2d(
    boxes1: Tensor,
    boxes2: Tensor,
    eps: float = 0,
) -> Tensor:
    """
    Computes the generalized box iou between given bounding boxes

    Args:
        boxes1: set of boxes (x1, y1, x2, y2)[N, 4]
        boxes2: set of boxes (x1, y1, x2, y2)[M, 4]
        eps: optional small constant for numerical stability

    Returns:
        Tensor: the NxM matrix containing the pairwise generalized IoU values
            for every element in boxes1 and boxes2, shape [N, M]

    Notes:
        always prefer using the n-D version since it takes care of data types.
    """
    iou, union = box_iou_union_2d(boxes1, boxes2)

    x1 = torch.min(boxes1[:, None, 0], boxes2[:, 0])  # [N, M]
    y1 = torch.min(boxes1[:, None, 1], boxes2[:, 1])  # [N, M]
    x2 = torch.max(boxes1[:, None, 2], boxes2[:, 2])  # [N, M]
    y2 = torch.max(boxes1[:, None, 3], boxes2[:, 3])  # [N, M]

    area = ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)) + eps  # [N, M]
    return iou - (area - union) / area


###
# Other Ops
###


def remove_small_boxes(
    boxes: Tensor,
    min_size: float,
) -> Tensor:
    """
    Remove boxes with at least one side smaller than min_size.

    Args:
        boxes: boxes (x1, y1, x2, y2, (z1, z2)) [N, dim * 2]
        min_size: minimum size

    Returns:
        Tensor: indices of the boxes that have all sides
            larger than min_size [N]
    """
    if boxes.shape[1] == 4:
        ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
    else:
        ws, hs, ds = (
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1],
            boxes[:, 5] - boxes[:, 4],
        )
        keep = (ws >= min_size) & (hs >= min_size) & (ds >= min_size)
    keep = torch.where(keep)[0]
    return keep


def box_center_dist(
    boxes1: Tensor,
    boxes2: Tensor,
    euclidean: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Distance of center points between two sets of boxes

    Args:
        boxes1: boxes (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        boxes2: boxes (x1, y1, x2, y2, (z1, z2))[M, dim * 2]
        euclidean: computed the euclidean distance otherwise it uses the l1
            distance

    Returns:
        Tensor: the NxM matrix containing the pairwise
            distances for every element in boxes1 and boxes2 [N, M]
        Tensor: center points of boxes1
        Tensor: center points of boxes2
    """
    center1 = box_center(boxes1)  # [N, dims]
    center2 = box_center(boxes2)  # [M, dims]

    if euclidean:
        dists = (center1[:, None] - center2[None]).pow(2).sum(-1).sqrt()
    else:
        # before sum: [N, M, dims]
        dists = (center1[:, None] - center2[None]).sum(-1)
    return dists, center1, center2


def center_in_boxes(
    center: Tensor,
    boxes: Tensor,
    eps: float = 0.01,
) -> Tensor:
    """
    Checks which center points are within boxes

    Args:
        center: center points [N, dims]
        boxes: boxes [N, dims * 2]
        eps: minimum distance to boarder of boxes

    Returns:
        Tensor: boolean array indicating which center points are within
            the boxes [N]
    """
    axes = []
    axes.append(center[:, 0] - boxes[:, 0])
    axes.append(center[:, 1] - boxes[:, 1])
    axes.append(boxes[:, 2] - center[:, 0])
    axes.append(boxes[:, 3] - center[:, 1])
    if center.shape[1] == 3:
        axes.append(center[:, 2] - boxes[:, 4])
        axes.append(boxes[:, 5] - center[:, 2])
    return torch.stack(axes, dim=1).min(dim=1)[0] > eps


def box_center(
    boxes: Tensor,
) -> Tensor:
    """
    Compute center point of boxes

    Args:
        boxes: bounding boxes (x1, y1, x2, y2, (z1, z2)) [N, dims * 2]

    Returns:
        Tensor: center points [N, dims]
    """
    centers = [(boxes[:, 2] + boxes[:, 0]) / 2.0, (boxes[:, 3] + boxes[:, 1]) / 2.0]
    if boxes.shape[1] == 6:
        centers.append((boxes[:, 5] + boxes[:, 4]) / 2.0)
    return torch.stack(centers, dim=1)


def permute_boxes(
    boxes: Union[Tensor, ndarray],
    dims: Sequence[int] = None,
) -> Union[Tensor, ndarray]:
    """
    Change ordering of axis of boxes

    Args:
        boxes: boxes [N, dims * 2](x1, y1, x2, y2(, z1, z2))
        dims: the desired ordering of dimensions; By default the dimensions
            are reversed

    Returns:
        Tensor: boxes with permuted axes [N, dims * 2]
    """
    if dims is None:
        dims = list(range(boxes.shape[1] // 2))[::-1]
    if 2 * len(dims) != boxes.shape[1]:
        raise TypeError(f"Need same number of dimensions, found dims {dims} " f"but boxes with shape {boxes.shape}")

    indexing = [[0, 2], [1, 3]]
    if boxes.shape[1] == 6:
        indexing.append([4, 5])
    new_axis = [
        indexing[dims[0]][0],
        indexing[dims[1]][0],
        indexing[dims[0]][1],
        indexing[dims[1]][1],
    ]
    for d in dims[2:]:
        new_axis.extend(indexing[d])
    return boxes[:, new_axis]


def expand_to_boxes(
    data: Tensor,
) -> Tensor:
    """
    Expand x,y,z data to box format

    Args:
        data: data to expand (N, dim)[:, (x, y, [z])]

    Returns:
        Tensor: expanded tensors
    """
    idx = [0, 1, 0, 1]
    if (data.ndim == 1 and data.shape[0] == 3) or (data.ndim == 2 and data.shape[1] == 3):
        idx.extend((2, 2))
    if data.ndim == 1:
        data = data[None]
    return data[:, idx]


def box_size(
    boxes: Tensor,
) -> Tensor:
    """
    Compute length of boxes along all dimensions

    Args:
        boxes: boxes (x1, y1, x2, y2, z1, z2)[N, dim * 2]

    Returns:
        Tensor: size along axis (x, y, (z))[N, dim]
    """
    dists = []
    dists.append(boxes[:, 2] - boxes[:, 0])
    dists.append(boxes[:, 3] - boxes[:, 1])
    if boxes.shape[1] // 2 == 3:
        dists.append(boxes[:, 5] - boxes[:, 4])
    return torch.stack(dists, axis=1)


def extend_and_cat_boxes(
    boxes: List[Tensor],
) -> Tensor:
    """
    Concatenate boxes of multiple images and add batch idx at first pos

    Args:
        boxes: sequence of boxes. (x1, y1, x2, y2, (z1, z2))[N, dim * 2]

    Returns:
        Tensor: concatenated boxes with batch index. The first index of each
            box corresponds to the batch idx
            (batch_idx, x1, y1, x2, y2, (z1, z2))[N, 1 + dim * 2]
    """
    extended_boxes = []
    for i, b in enumerate(boxes):
        extended_boxes.append(
            torch.cat(
                [torch.full((b.shape[0], 1), i, dtype=b.dtype, device=b.device), b],
                dim=1,
            )
        )
    return torch.cat(extended_boxes, dim=0)


def cat_and_index(
    boxes: List[Tensor],
) -> Tuple[Tensor, Tensor]:
    indices = []
    for i, b in enumerate(boxes):
        indices.append(torch.full((b.shape[0],), i, dtype=b.dtype, device=b.device))
    return torch.cat(boxes, dim=0), torch.cat(indices, dim=0)


def clip_boxes_to_image_(
    boxes: torch.Tensor,
    img_shape: Tuple[int],
):
    """
    Clip boxes to image dimensions inplace

    Args:
        boxes: tensor with boxes [N x (2*dim)]
            (x_min, y_min, x_max, y_max(, z_min, z_max))
        img_shape: size of image

    Returns:
        Tensor: clipped boxes as tensor

    Raises:
        ValueError: boxes need to have 4(2D) or 6(3D) components
    """
    if boxes.shape[-1] == 4:
        return clip_boxes_to_image_2d_(boxes, img_shape)
    elif boxes.shape[-1] == 6:
        return clip_boxes_to_image_3d_(boxes, img_shape)
    else:
        raise ValueError(f"Boxes with {boxes.shape[-1]} are not supported.")


def clip_boxes_to_image(
    boxes: torch.Tensor,
    img_shape: Tuple[int],
):
    """
    Clip boxes to image dimensions

    Args:
        boxes: tensor with boxes [N x (2*dim)]
            (x_min, y_min, x_max, y_max(, z_min, z_max))
        img_shape: size of image

    Returns:
        Tensor: clipped boxes as tensor

    Raises:
        ValueError: boxes need to have 4(2D) or 6(3D) components
    """
    if boxes.shape[-1] == 4:
        return clip_boxes_to_image_2d(boxes, img_shape)
    elif boxes.shape[-1] == 6:
        return clip_boxes_to_image_3d(boxes, img_shape)
    else:
        raise ValueError(f"Boxes with {boxes.shape[-1]} are not supported.")


def clip_boxes_to_image_2d_(
    boxes: torch.Tensor,
    img_shape: Tuple[int, int],
):
    """
    Clip boxes to image dimensions

    Args:
        boxes: tensor with boxes [N x 4] (x_min, y_min, x_max, y_max)
        img_shape: size of image

    Returns:
        Tensor: clipped boxes as tensor
    """
    s0, s1 = img_shape
    boxes[..., 0::2].clamp_(min=-1, max=s0)
    boxes[..., 1::2].clamp_(min=-1, max=s1)
    return boxes


def clip_boxes_to_image_3d_(
    boxes: torch.Tensor,
    img_shape: Tuple[int, int, int],
):
    """
    Clip boxes to image dimensions

    Args:
        boxes: tensor with boxes [N x 6]
            (x_min, y_min, x_max, y_max, z_min, z_max)
        img_shape: size of image

    Returns:
        Tensor: clipped boxes as tensor
    """
    s0, s1, s2 = img_shape
    boxes[..., 0::6].clamp_(min=-1, max=s0)
    boxes[..., 1::6].clamp_(min=-1, max=s1)
    boxes[..., 2::6].clamp_(min=-1, max=s0)
    boxes[..., 3::6].clamp_(min=-1, max=s1)
    boxes[..., 4::6].clamp_(min=-1, max=s2)
    boxes[..., 5::6].clamp_(min=-1, max=s2)
    return boxes


def clip_boxes_to_image_2d(
    boxes: torch.Tensor,
    img_shape: Tuple[int, int],
):
    """
    Clip boxes to image dimensions

    Args:
        boxes: tensor with boxes [N x 4] (x_min, y_min, x_max, y_max)
        img_shape: size of image

    Returns:
        Tensor: clipped boxes as tensor

    Notes:
        Uses float32 internally because clipping of half cpu tensors is not
        supported
    """
    s0, s1 = img_shape
    boxes[..., 0::2] = boxes[..., 0::2].clamp(min=-1, max=s0)
    boxes[..., 1::2] = boxes[..., 1::2].clamp(min=-1, max=s1)
    return boxes


def clip_boxes_to_image_3d(
    boxes: torch.Tensor,
    img_shape: Tuple[int, int, int],
):
    """
    Clip boxes to image dimensions

    Args:
        boxes: tensor with boxes [N x 6]
            (x_min, y_min, x_max, y_max, z_min, z_max)
        img_shape: size of image

    Returns:
        Tensor: clipped boxes as tensor

    Notes:
        Uses float32 internally because clipping of half cpu tensors is not
        supported
    """
    s0, s1, s2 = img_shape
    boxes[..., 0::6] = boxes[..., 0::6].clamp(min=-1, max=s0)
    boxes[..., 1::6] = boxes[..., 1::6].clamp(min=-1, max=s1)
    boxes[..., 2::6] = boxes[..., 2::6].clamp(min=-1, max=s0)
    boxes[..., 3::6] = boxes[..., 3::6].clamp(min=-1, max=s1)
    boxes[..., 4::6] = boxes[..., 4::6].clamp(min=-1, max=s2)
    boxes[..., 5::6] = boxes[..., 5::6].clamp(min=-1, max=s2)
    return boxes


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
        return torch.tensor([], device=masks.device, dtype=masks.dtype)

    boxes_size = torch.round(box_size(boxes)).to(dtype=torch.int)
    for idx in range(num_items):
        if (boxes_size[idx] < 1).any():
            continue
        _mask_rescale = F.interpolate(
            masks[idx][None, None],
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
            image_coords.append(slice(int(boxes[idx, 4]), int(boxes[idx, 4]) + int(boxes_size[idx, 2])))
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


def inverse_sigmoid(data: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Inverse Sigmoid Function for pre-sigmoid additions

    Args:
        data: input tensor
        eps: epsilon for numerical stability

    Returns:
        torch.Tensor: inverse sigmoid of values in original tensor
    """
    data = data.clamp(min=0, max=1)
    x1 = data.clamp(min=eps)
    x2 = (1 - data).clamp(min=eps)
    return torch.log(x1 / x2)


def box_point_norm_with_size(
    boxes: torch.Tensor,
    img_shape: ND_TUPLE_INT,
) -> torch.Tensor:
    """
    Normalize boxes in point format with image size (range will be 0,1)

    Args:
        boxes: bounding boxes [x0, y0, x1, y1 (, z0, z1)] with shape
            [N, dim * 2]
        img_shape: shape of input (in training, this corresponds to the shape
            of the reference frame of the box, usually the extracted patch)

    Returns:
        torch.Tensor: normalized boxes [x0, y0, x1, y1 (, z0, z1)]
    """
    if boxes.numel() == 0:  # handle empty boxes
        return boxes

    img_shape_tensor = torch.tensor(img_shape, dtype=boxes.dtype, device=boxes.device)
    return boxes / expand_to_boxes(img_shape_tensor[None])


def box_point_rescale_with_size(
    boxes: torch.Tensor,
    img_shape: ND_TUPLE_INT,
    extra_batched: bool = False,
) -> torch.Tensor:
    """
    Revert normalization from boxes in point format with image size.

    Args:
        boxes: bounding boxes [x0, y0, x1, y1 (, z0, z1)] with shape
            [N, dim * 2]
        img_shape: shape of input (in training, this corresponds to the shape
            of the reference frame of the box, usually the extracted patch)
        extra_batched: provided bounding boxes are in format [B, R, dims * 2]

    Returns:
        torch.Tensor: rescaled boxes [x0, y0, x1, y1 (, z0, z1)]
    """
    if boxes.numel() == 0:  # handle empty boxes
        return boxes

    img_shape_tensor = torch.tensor(img_shape, dtype=boxes.dtype, device=boxes.device)
    if extra_batched:
        return boxes * expand_to_boxes(img_shape_tensor[None])[None]
    else:
        return boxes * expand_to_boxes(img_shape_tensor[None])


def box_point2center_format(boxes_point: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from point [x0, y0, x1, y1 (, z0, z1)] into
    center format [cx, cy, dx, dy (, cz, dz)]

    Args:
        boxes_point: input boxes in format [x0, y0, x1, y1 (, z0, z1)] with
            shape [*, dim * 2]

    Returns:
        torch.Tensor: boxes in format [cx, cy, dx, dy (, cz, dz)] with shape
            [*, dim * 2]
    """
    if boxes_point.numel() == 0:  # handle empty boxes
        return boxes_point

    if boxes_point.shape[-1] == 4:
        x0, y0, x1, y1 = boxes_point.unbind(-1)
        bc = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    else:
        x0, y0, x1, y1, z0, z1 = boxes_point.unbind(-1)
        bc = [
            (x0 + x1) / 2,
            (y0 + y1) / 2,
            (x1 - x0),
            (y1 - y0),
            (z0 + z1) / 2,
            (z1 - z0),
        ]
    return torch.stack(bc, dim=-1)


def box_center2point_format(boxes_center: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from center [cx, cy, dx, dy (, cz, dz)] into
    point format [x0, y0, x1, y1 (, z0, z1)]

    Args:
        boxes_point: input boxes in format [cx, cy, dx, dy (, cz, dz)] with
            shape [*, dim * 2]

    Returns:
        torch.Tensor: boxes in format [x0, y0, x1, y1 (, z0, z1)] with shape
            [*, dim * 2]
    """
    if boxes_center.numel() == 0:  # handle empty boxes
        return boxes_center

    if boxes_center.shape[-1] == 4:
        cx, cy, dx, dy = boxes_center.unbind(-1)
        bp = [
            cx - 0.5 * dx,
            cy - 0.5 * dy,
            cx + 0.5 * dx,
            cy + 0.5 * dy,
        ]
    else:
        cx, cy, dx, dy, cz, dz = boxes_center.unbind(-1)
        bp = [
            cx - 0.5 * dx,
            cy - 0.5 * dy,
            cx + 0.5 * dx,
            cy + 0.5 * dy,
            cz - 0.5 * dz,
            cz + 0.5 * dz,
        ]
    return torch.stack(bp, dim=-1)
