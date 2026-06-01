# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from itertools import product
from typing import List, Sequence

import numpy as np
from numpy import ndarray

from nndet.core.ops_torch import expand_to_boxes as _expand_to_boxes
from nndet.utils.tensor import ensure_min_float32_np
from nndet.utils.typing import ND_TUPLE_INT


def expand_to_boxes(
    data: ndarray,
) -> ndarray:
    """
    Expand x,y,z data to box format

    Args:
        data: data to expand (N, dim)[:, (x, y, [z])]

    Returns:
        ndarray: expanded tensors
    """
    return _expand_to_boxes(data)


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


def clip_boxes_to_image(
    boxes: np.ndarray,
    img_shape: ND_TUPLE_INT,
) -> np.ndarray:
    """
    Clip boxes to image dimensions

    Args:
        boxes: array with boxes [N x (2*dim)]
            (x_min, y_min, x_max, y_max(, z_min, z_max))
        img_shape: size of image

    Returns:
        np.ndarray: clipped boxes as tensor

    Raises:
        ValueError: boxes need to have 4(2D) or 6(3D) components
    """
    if boxes.shape[-1] == 4:
        return clip_boxes_to_image_2d(boxes, img_shape)
    elif boxes.shape[-1] == 6:
        return clip_boxes_to_image_3d(boxes, img_shape)
    else:
        raise ValueError(f"Boxes with {boxes.shape[-1]} are not supported.")


def clip_boxes_to_image_2d(
    boxes: np.ndarray,
    img_shape: ND_TUPLE_INT,
) -> np.ndarray:
    """
    Clip boxes to image dimensions

    Args:
        boxes: array with boxes [N x 4] (x_min, y_min, x_max, y_max)
        img_shape: size of image

    Returns:
        Tensor: clipped boxes as array

    Notes:
        Uses float32 internally because clipping of half cpu tensors is not
        supported
    """
    s0, s1 = img_shape
    boxes[..., 0::2] = np.clip(boxes[..., 0::2], a_min=-1, a_max=s0)
    boxes[..., 1::2] = np.clip(boxes[..., 1::2], a_min=-1, a_max=s1)
    return boxes


def clip_boxes_to_image_3d(
    boxes: np.ndarray,
    img_shape: ND_TUPLE_INT,
) -> np.ndarray:
    """
    Clip boxes to image dimensions

    Args:
        boxes: array with boxes [N x 6]
            (x_min, y_min, x_max, y_max, z_min, z_max)
        img_shape: size of image

    Returns:
        Tensor: clipped boxes as array

    Notes:
        Uses float32 internally because clipping of half cpu tensors is not
        supported
    """
    s0, s1, s2 = img_shape
    boxes[..., 0::6] = np.clip(boxes[..., 0::6], a_min=-1, a_max=s0)
    boxes[..., 1::6] = np.clip(boxes[..., 1::6], a_min=-1, a_max=s1)
    boxes[..., 2::6] = np.clip(boxes[..., 2::6], a_min=-1, a_max=s0)
    boxes[..., 3::6] = np.clip(boxes[..., 3::6], a_min=-1, a_max=s1)
    boxes[..., 4::6] = np.clip(boxes[..., 4::6], a_min=-1, a_max=s2)
    boxes[..., 5::6] = np.clip(boxes[..., 5::6], a_min=-1, a_max=s2)
    return boxes


def remove_small_boxes(
    boxes: np.ndarray,
    min_size: float,
) -> np.ndarray:
    """
    Remove boxes with at least one side smaller than min_size.

    Args:
        boxes: boxes (x1, y1, x2, y2, (z1, z2)) [N, dim * 2]
        min_size: minimum size

    Returns:
        array: indices of the boxes that have all sides
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
    keep = np.nonzero(keep)[0]
    return keep


# Point Operations


def points_to_homogeneous(points: Sequence[np.ndarray]) -> List[np.ndarray]:
    """
    Transforms points from cartesian to homogeneous coordinates

    Args:
        points: list of points to transform List([*, #dims]) where * are arbitrary
            dimensions and dims is the number of spatial dimensions

    Returns:
        List[np.ndarray]: the batch of points in homogeneous
            coordinates [*, #dim + 1]
    """
    return [np.concatenate([p, np.ones((*p.shape[:-1], 1), dtype=p.dtype)], axis=-1) for p in points]


def points_to_cartesian(points: Sequence[np.ndarray]) -> List[np.ndarray]:
    """
    Transforms points in homogeneous coordinates back to cartesian
    coordinates.

    Args:
        points: homogeneous points List([N, in_dims]), N number of points,
            in_dims number of input dimensions (spatial dimensions + 1)

    Returns:
        List[np.ndarray]: cartesian points [N, in_dims] = [N, dims]
    """
    return [p[..., :-1] / p[..., -1][..., None] for p in points]


def boxes2corner_points(boxes: np.ndarray) -> ndarray:
    """
    Convert boxes to corner points

    Args:
        boxes: boxes of shape [N, dims] where is the number of boxes and dims
            is the number of spatial dimensions

    Returns:
        ndarray: corner points [N, 2 ** #dims, dims], where N is the
            number of boxes and #dims is the number of spatial dimensions
    """
    if boxes.size == 0:
        return np.array([[[]]]).reshape(0, 2 ** (boxes.shape[1] // 2), boxes.shape[1] // 2)

    if boxes.shape[1] == 4:
        idx = list(product([0, 2], [1, 3]))
        corners = np.stack([np.stack([boxes[:, i[0]], boxes[:, i[1]]], axis=-1) for i in idx], axis=1)
    elif boxes.shape[1] == 6:
        idx = list(product([0, 2], [1, 3], [4, 5]))
        corners = np.stack(
            [np.stack([boxes[:, i[0]], boxes[:, i[1]], boxes[:, i[2]]], axis=-1) for i in idx],
            axis=1,
        )
    else:
        raise ValueError(f"Unsupported dimensionality of boxes, found {boxes.ndim} dimensions")
    return corners


def boxes2center_points(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes to the center of area points

    Args:
        boxes: boxes of shape [N, dims] where is the number of boxes and dims
            is the number of spatial dimensions

    Returns:
        ndarray: corner points [N, 2 ** #dims, dims], where N is the
            number of boxes and #dims is the number of spatial dimensions
    """
    if boxes.size == 0:
        return np.array([[[]]]).reshape(0, 2 ** (boxes.shape[1] // 2), boxes.shape[1] // 2)

    dim = boxes.shape[-1] // 2
    num_obj = boxes.shape[0]
    assert dim in [2, 3]
    center = box_center_np(boxes)  # N, #dims
    size = box_size_np(boxes)  # N, #dims

    points = []
    for d in range(dim):
        offset = np.zeros_like(center)  # N, #dims
        np.put_along_axis(offset, np.array([[d]] * num_obj), size[:, d : d + 1] / 2, axis=1)
        points.append(center + offset)
        points.append(center - offset)
    return np.stack(points, axis=1)  # N, P, #dims


def object_points2boxes(points: np.ndarray) -> np.ndarray:
    """
    Convert unordered set of points to boxes

    Args:
        points: points of shape [N, P, #dims] where N is the number of objects,
            P is the number of points per object and #dims is the number of
            spatial dimensions

    Returns:
        ndarray: boxes of shape [N, #dims] where is the number of boxes and
            #dims is the number of spatial dimensions
    """
    if points.shape[-1] not in [2, 3]:
        raise ValueError("Only support 2 and 3 dimensional points")

    if points.size > 0:
        t = [a[..., 0] for a in np.split(points, points.shape[-1], -1)]
        points_ax0, points_ax1 = t[0], t[1]  # [N, P]
        if points.shape[-1] == 3:
            points_ax2 = t[2]  # [N, P]

        boxes = np.zeros((points.shape[0], points.shape[-1] * 2), dtype=points.dtype)
        boxes[:, 0] = np.min(points_ax0, axis=1)
        boxes[:, 1] = np.min(points_ax1, axis=1)
        boxes[:, 2] = np.max(points_ax0, axis=1)
        boxes[:, 3] = np.max(points_ax1, axis=1)
        if points.shape[-1] == 3:
            boxes[:, 4] = np.min(points_ax2, axis=1)
            boxes[:, 5] = np.max(points_ax2, axis=1)
    else:
        boxes = np.tensor([]).reshape(-1, points.shape[-1] * 2, dtype=points.dtype)
    return boxes


# These are specialized transformations for the mirror augmentation TTA
# Prefer using the functions above for point transformations


def boxes2points(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes to 2 points

    Args:
        boxes: (x1, y1, x2, y2, (z1, z2))[N, dims x 2]

    Returns:
        np.ndarray: points [N x 2, dims]
    """
    if boxes.shape[1] == 4:
        idx0 = [0, 1]
        idx1 = [2, 3]
    else:
        idx0 = [0, 1, 4]
        idx1 = [2, 3, 5]

    points0 = boxes[:, idx0]
    points1 = boxes[:, idx1]
    return np.concatenate([points0, points1], axis=0)


def points2boxes(points: np.ndarray) -> np.ndarray:
    """
    Convert 2 points to boxes

    Args:
        points: boxes need to be order as specified
            order: [point_box_0, ... point_box_N/2] * 4
            format of points: (x, y(, z)))[N, dims]

    Returns:
        np.ndarray: bounding boxes [N / 2, dims * 2]
    """
    if points.nelement() > 0:
        points0, points1 = points.split(points.shape[0] // 2)
        boxes = np.zeros((points.shape[0] // 2, points.shape[1] * 2), dtype=points.dtype)
        boxes[:, 0] = np.min(points0[:, 0], points1[:, 0])
        boxes[:, 1] = np.min(points0[:, 1], points1[:, 1])
        boxes[:, 2] = np.max(points0[:, 0], points1[:, 0])
        boxes[:, 3] = np.max(points0[:, 1], points1[:, 1])
        if boxes.shape[1] == 6:
            boxes[:, 4] = np.min(points0[:, 2], points1[:, 2])
            boxes[:, 5] = np.max(points0[:, 2], points1[:, 2])
        return boxes
    else:
        return np.tensor([]).reshape(-1, points.shape[1] * 2, dtype=points.dtype)
