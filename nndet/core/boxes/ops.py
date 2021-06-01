"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch

from torch import Tensor
from numpy import ndarray
from typing import Union, Sequence, Tuple


from torch.cuda.amp import autocast


def box_area_3d(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2, z1, z2) coordinates.
    
    Arguments:
        boxes (Union[Tensor, ndarray]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2, z1, z2) format. [N, 6]
    Returns:
        area (Union[Tensor, ndarray]): area for each box [N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 4])


def box_area_2d(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Union[Tensor, ndarray]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format. [N, 4]
    Returns:
        area (Union[Tensor, ndarray]): area for each box [N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_area(boxes: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
    """
    Computes the area of a set of bounding boxes
    
    Args:
        boxes (Union[Tensor, ndarray]): boxes of shape; (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
    
    Returns:
        Union[Tensor, ndarray]: area of boxes
    
    See Also:
        :func:`box_area_3d`, :func:`torchvision.ops.boxes.box_area`
    """
    if boxes.shape[-1] == 4:
        return box_area_2d(boxes)
    else:
        return box_area_3d(boxes)


@autocast(enabled=False)
def box_iou(boxes1: Tensor, boxes2: Tensor,  eps: float = 0) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) of boxes.
    (Works for Tensors and Numpy Arrays)

    Arguments:
        boxes1: boxes; (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        boxes2: boxes; (x1, y1, x2, y2, (z1, z2))[M, dim * 2]
        eps: optional small constant for numerical stability

    Returns:
        iou (Tensor): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2; [N, M]

    See Also:
        :func:`box_iou_3d`, :func:`torchvision.ops.boxes.box_iou`

    Notes:
        Need to compute IoU in float32 (autocast=False) because the
        volume/area can be to large
    """
    # TODO: think about adding additional assert statements to check coordinates x1 <= x2, y1 <= y2, z1 <= z2
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.tensor([]).to(boxes1)
    if boxes1.shape[-1] == 4:
        return box_iou_union_2d(boxes1.float(), boxes2.float(), eps=eps)[0]
    else:
        return box_iou_union_3d(boxes1.float(), boxes2.float(), eps=eps)[0]


@autocast(enabled=False)
def generalized_box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 0) -> Tensor:
    """
    Generalized box iou

    Arguments:
        boxes1: boxes; (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        boxes2: boxes; (x1, y1, x2, y2, (z1, z2))[M, dim * 2]
        eps: optional small constant for numerical stability

    Returns:
        Tensor: the NxM matrix containing the pairwise
            generalized IoU values for every element in boxes1 and boxes2; [N, M]

    Notes:
        Need to compute IoU in float32 (autocast=False) because the
        volume/area can be to large
    """
    if boxes1.nelement() == 0 or boxes2.nelement() == 0:
        return torch.tensor([]).to(boxes1)
    if boxes1.shape[-1] == 4:
        return generalized_box_iou_2d(boxes1.float(), boxes2.float(), eps=eps)
    else:
        return generalized_box_iou_3d(boxes1.float(), boxes2.float(), eps=eps)


def box_iou_union_3d(boxes1: Tensor, boxes2: Tensor, eps: float = 0) -> Tuple[Tensor, Tensor]:
    """
    Return intersection-over-union (Jaccard index) and  of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2, z1, z2) format.
    
    Args:
        boxes1: set of boxes (x1, y1, x2, y2, z1, z2)[N, 6]
        boxes2: set of boxes (x1, y1, x2, y2, z1, z2)[M, 6]
        eps: optional small constant for numerical stability

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
        Tensor[N, M]: the nxM matrix containing the pairwise union
            values
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
    union = (vol1[:, None] + vol2 - inter)
    return inter / union, union


def generalized_box_iou_3d(boxes1: Tensor, boxes2: Tensor, eps: float = 0) -> Tensor:
    """
    Computes the generalized box iou between given bounding boxes

    Args:
        boxes1: set of boxes (x1, y1, x2, y2, z1, z2)[N, 6]
        boxes2: set of boxes (x1, y1, x2, y2, z1, z2)[M, 6]
        eps: optional small constant for numerical stability

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise
            generalized IoU values for every element in boxes1 and boxes2
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


def box_iou_union_2d(boxes1: Tensor, boxes2: Tensor, eps: float = 0) -> Tuple[Tensor, Tensor]:
    """
    Return intersection-over-union (Jaccard index) and  of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1: set of boxes (x1, y1, x2, y2)[N, 4]
        boxes2: set of boxes (x1, y1, x2, y2)[M, 4]
        eps: optional small constant for numerical stability

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
        union (Tensor[N, M]): the nxM matrix containing the pairwise union
            values
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # [N, M]
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])  # [N, M]
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])  # [N, M]
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])  # [N, M]

    inter = ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)) + eps  # [N, M]
    union = (area1[:, None] + area2 - inter)
    return inter / union, union


def generalized_box_iou_2d(boxes1: Tensor, boxes2: Tensor, eps: float = 0) -> Tensor:
    """
    Computes the generalized box iou between given bounding boxes

    Args:
        boxes1: set of boxes (x1, y1, x2, y2)[N, 4]
        boxes2: set of boxes (x1, y1, x2, y2)[M, 4]
        eps: optional small constant for numerical stability

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise
            generalized IoU values for every element in boxes1 and boxes2
    """
    iou, union = box_iou_union_2d(boxes1, boxes2)

    x1 = torch.min(boxes1[:, None, 0], boxes2[:, 0])  # [N, M]
    y1 = torch.min(boxes1[:, None, 1], boxes2[:, 1])  # [N, M]
    x2 = torch.max(boxes1[:, None, 2], boxes2[:, 2])  # [N, M]
    y2 = torch.max(boxes1[:, None, 3], boxes2[:, 3])  # [N, M]

    area = ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)) + eps  # [N, M]
    return iou - (area - union) / area


def remove_small_boxes(boxes: Tensor, min_size: float) -> Tensor:
    """
    Remove boxes with at least one side smaller than min_size.
    
    Arguments:
        boxes (Tensor): boxes (x1, y1, x2, y2, (z1, z2)) [N, dim * 2]
        min_size (float): minimum size
    Returns:
        keep (Tensor): indices of the boxes that have both sides
            larger than min_size [N]
    """
    if boxes.shape[1] == 4:
        ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
    else:
        ws, hs, ds = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1], boxes[:, 5] - boxes[:, 4]
        keep = (ws >= min_size) & (hs >= min_size) & (ds >= min_size)
    keep = torch.where(keep)[0]
    return keep


def box_center_dist(boxes1: Tensor, boxes2: Tensor, euclidean: bool = True) -> \
        Tuple[Tensor, Tensor, Tensor]:
    """
    Distance of center points between two sets of boxes

    Arguments:
        boxes1: boxes; (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        boxes2: boxes; (x1, y1, x2, y2, (z1, z2))[M, dim * 2]
        euclidean: computed the euclidean distance otherwise it uses the l1
            distance

    Returns:
        Tensor: the NxM matrix containing the pairwise
            distances for every element in boxes1 and boxes2; [N, M]
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


def center_in_boxes(center: Tensor, boxes: Tensor, eps: float = 0.01) -> Tensor:
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


def box_center(boxes: Tensor) -> Tensor:
    """
    Compute center point of boxes

    Args:
        boxes: bounding boxes (x1, y1, x2, y2, (z1, z2)) [N, dims * 2]

    Returns:
        Tensor: center points [N, dims]
    """
    centers = [(boxes[:, 2] + boxes[:, 0]) / 2., (boxes[:, 3] + boxes[:, 1]) / 2.]
    if boxes.shape[1] == 6:
        centers.append((boxes[:, 5] + boxes[:, 4]) / 2.)
    return torch.stack(centers, dim=1)


def permute_boxes(boxes: Union[Tensor, ndarray],
                  dims: Sequence[int] = None) -> Union[Tensor, ndarray]:
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
        raise TypeError(f"Need same number of dimensions, found dims {dims} "
                        f"but boxes with shape {boxes.shape}")

    indexing = [[0, 2], [1, 3]]
    if boxes.shape[1] == 6:
        indexing.append([4, 5])
    new_axis = [indexing[dims[0]][0], indexing[dims[1]][0],
                indexing[dims[0]][1], indexing[dims[1]][1]]
    for d in dims[2:]:
        new_axis.extend(indexing[d])
    return boxes[:, new_axis]


def expand_to_boxes(data: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
    """
    Expand x,y,z data to box format
    
    Args:
        data (Tensor): data to expand (N, dim)[:, (x, y, [z])]
    
    Returns:
        Tensor: expanded tensors
    """
    idx = [0, 1, 0, 1]
    if (len(data.shape) == 1 and data.shape[0] == 3) or (len(data.shape) == 2 and data.shape[1] == 3):
        idx.extend((2, 2))
    if len(data.shape) == 1:
        data = data[None]
    return data[:, idx]


def box_size(boxes: Tensor) -> Tensor:
    """
    Compute length of boxes along all dimensions
    
    Args:
        boxes (Tensor): boxes (x1, y1, x2, y2, z1, z2)[N, dim * 2]
    
    Returns:
        Tensor: size along axis (x, y, (z))[N, dim]
    """
    dists = []
    dists.append(boxes[:, 2] - boxes[:, 0])
    dists.append(boxes[:, 3] - boxes[:, 1])
    if boxes.shape[1] // 2 == 3:
        dists.append(boxes[:, 5] - boxes[:, 4])
    return torch.stack(dists, axis=1)
