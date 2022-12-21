# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy import ndarray

from nndet.utils.tensor import ensure_min_float32_np


def box_area_np(
    boxes: ndarray,
) -> ndarray:
    """
    Notes:
        always prefer using the n-D version since it takes care of data types.

    See Also:
        :func:`nndet.core.boxes.ops.box_area`
    """
    _boxes = ensure_min_float32_np(boxes)
    if boxes.shape[-1] == 4:
        return box_area_2d_np(_boxes)
    else:
        return box_area_3d_np(_boxes)


def box_area_3d_np(
    boxes: np.ndarray,
) -> np.ndarray:
    """
    Notes:
        always prefer using the n-D version since it takes care of data types.

    See Also:
        `nndet.core.boxes.ops.box_area_3d`
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 4])


def box_area_2d_np(
    boxes: np.ndarray,
) -> np.ndarray:
    """
    Notes:
        always prefer using the n-D version since it takes care of data types.

    See Also:
        `nndet.core.boxes.ops.box_area_2d`
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou_np(
    boxes1: ndarray,
    boxes2: ndarray,
) -> ndarray:
    """
    Return intersection-over-union (Jaccard index) of boxes.
    (Works for ndarrays and Numpy Arrays)

    Args:
        boxes1: boxes (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        boxes2: boxes (x1, y1, x2, y2, (z1, z2))[M, dim * 2]

    Returns:
        ndarray: the NxM iou matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2; shape [N, M]

    See Also:
        :func:`box_iou_3d`, :func:`torchvision.ops.boxes.box_iou`
    """
    _boxes1 = ensure_min_float32_np(boxes1)
    _boxes2 = ensure_min_float32_np(boxes2)
    if boxes1.shape[-1] == 4:
        return box_iou_2d_np(_boxes1, _boxes2)
    else:
        return box_iou_3d_np(_boxes1, _boxes2)


def box_iou_2d_np(
    boxes1: ndarray,
    boxes2: ndarray,
) -> ndarray:
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Args:
        boxes1: set of boxes (x1, y1, x2, y2)[N, 4]
        boxes2: set of boxes (x1, y1, x2, y2)[M, 4]

    Returns:
        ndarray: the NxM iou matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2

    Notes:
        always prefer using the n-D version since it takes care of data types.
    """
    area1 = box_area_2d_np(boxes1)
    area2 = box_area_2d_np(boxes2)

    x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])  # [N, M]
    y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])  # [N, M]
    x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])  # [N, M]
    y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])  # [N, M]

    inter = np.clip((x2 - x1), a_min=0, a_max=None) * np.clip((y2 - y1), a_min=0, a_max=None)  # [N, M]
    return inter / (area1[:, None] + area2 - inter)


def box_iou_3d_np(
    boxes1: ndarray,
    boxes2: ndarray,
) -> ndarray:
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2, z1, z2) format.

    Args:
        boxes1: set of boxes (x1, y1, x2, y2, z1, z2)[N, 6]
        boxes2: set of boxes (x1, y1, x2, y2, z1, z2)[M, 6]

    Returns:
        ndarray: the NxM iou matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2

    Notes:
        always prefer using the n-D version since it takes care of data types.
    """
    area1 = box_area_3d_np(boxes1)
    area2 = box_area_3d_np(boxes2)

    x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])  # [N, M]
    y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])  # [N, M]
    x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])  # [N, M]
    y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])  # [N, M]
    z1 = np.maximum(boxes1[:, None, 4], boxes2[:, 4])  # [N, M]
    z2 = np.minimum(boxes1[:, None, 5], boxes2[:, 5])  # [N, M]

    inter = (
        np.clip((x2 - x1), a_min=0, a_max=None)
        * np.clip((y2 - y1), a_min=0, a_max=None)
        * np.clip((z2 - z1), a_min=0, a_max=None)
    )  # [N, M]
    return inter / (area1[:, None] + area2 - inter)


def box_size_np(
    boxes: ndarray,
) -> ndarray:
    """
    Compute length of boxes along all dimensions

    Args:
        boxes: boxes (x1, y1, x2, y2, z1, z2)[N, dim * 2]

    Returns:
        ndarray: size along axis (x, y, (z))[N, dim]
    """
    dists = []
    dists.append(boxes[:, 2] - boxes[:, 0])
    dists.append(boxes[:, 3] - boxes[:, 1])
    if boxes.shape[1] // 2 == 3:
        dists.append(boxes[:, 5] - boxes[:, 4])
    return np.stack(dists, axis=-1)


def box_center_np(
    boxes: np.ndarray,
) -> np.ndarray:
    """
    Compute center point of boxes

    Args:
        boxes: bounding boxes (x1, y1, x2, y2, (z1, z2)) [N, dims * 2]

    Returns:
        ndarray: center points [N, dims]
    """
    centers = [(boxes[:, 2] + boxes[:, 0]) / 2.0, (boxes[:, 3] + boxes[:, 1]) / 2.0]
    if boxes.shape[1] == 6:
        centers.append((boxes[:, 5] + boxes[:, 4]) / 2.0)
    return np.stack(centers, axis=1)


def bin_mask_iou_np(
    bin_masks1: np.ndarray,
    bin_masks2: np.ndarray,
) -> np.ndarray:
    bin_masks1_flattened = bin_masks1.reshape(bin_masks1.shape[0], -1)
    bin_masks2_flattened = bin_masks2.reshape(bin_masks2.shape[0], -1)

    masks1_vol = bin_masks1_flattened.sum(axis=1)  # [N]
    masks2_vol = bin_masks2_flattened.sum(axis=1)  # [M]

    intersection = np.matmul(bin_masks1_flattened, bin_masks2_flattened.T)  # [N, M]
    union = masks1_vol[:, None] + masks2_vol[None] - intersection  # [N, M]
    return intersection / union
