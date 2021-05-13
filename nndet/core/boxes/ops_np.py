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

import numpy as np
from numpy import ndarray


def box_area_np(boxes: ndarray) -> ndarray:
    """
    See Also:
        :func:`nndet.core.boxes.ops.box_area`
    """
    if boxes.shape[-1] == 4:
        return box_area_2d_np(boxes)
    else:
        return box_area_3d_np(boxes)


def box_area_3d_np(boxes: np.ndarray) -> np.ndarray:
    """
    See Also:
        `nndet.core.boxes.ops.box_area_3d`
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 4])


def box_area_2d_np(boxes: np.ndarray) -> np.ndarray:
    """
    See Also:
        `nndet.core.boxes.ops.box_area_2d`
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou_np(boxes1: ndarray, boxes2: ndarray) -> ndarray:
    """
    Return intersection-over-union (Jaccard index) of boxes.
    (Works for ndarrays and Numpy Arrays)

    Arguments:
        boxes1 (ndarray): boxes; (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        boxes2 (ndarray): boxes; (x1, y1, x2, y2, (z1, z2))[M, dim * 2]

    Returns:
        iou (ndarray): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2; [N, M]

    See Also:
        :func:`box_iou_3d`, :func:`torchvision.ops.boxes.box_iou`
    """
    # TODO: think about adding additional assert statements to check coordinates x1 <= x2, y1 <= y2, z1 <= z2
    if boxes1.shape[-1] == 4:
        return box_iou_2d_np(boxes1, boxes2)
    else:
        return box_iou_3d_np(boxes1, boxes2)


def box_iou_2d_np(boxes1: ndarray, boxes2: ndarray) -> ndarray:
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (ndarray): set of boxes (x1, y1, x2, y2)[N, 4]
        boxes2 (ndarray): set of boxes (x1, y1, x2, y2)[M, 4]

    Returns:
        iou (ndarray[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area_2d_np(boxes1)
    area2 = box_area_2d_np(boxes2)

    x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])  # [N, M]
    y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])  # [N, M]
    x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])  # [N, M]
    y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])  # [N, M]

    inter = np.clip((x2 - x1), a_min=0, a_max=None) * np.clip((y2 - y1), a_min=0, a_max=None)  # [N, M]
    return inter / (area1[:, None] + area2 - inter)


def box_iou_3d_np(boxes1: ndarray, boxes2: ndarray) -> ndarray:
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2, z1, z2) format.

    Arguments:
        boxes1 (ndarray): set of boxes (x1, y1, x2, y2, z1, z2)[N, 6]
        boxes2 (ndarray): set of boxes (x1, y1, x2, y2, z1, z2)[M, 6]

    Returns:
        iou (ndarray[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area_3d_np(boxes1)
    area2 = box_area_3d_np(boxes2)

    x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])  # [N, M]
    y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])  # [N, M]
    x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])  # [N, M]
    y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])  # [N, M]
    z1 = np.maximum(boxes1[:, None, 4], boxes2[:, 4])  # [N, M]
    z2 = np.minimum(boxes1[:, None, 5], boxes2[:, 5])  # [N, M]

    inter = np.clip((x2 - x1), a_min=0, a_max=None) * np.clip((y2 - y1), a_min=0, a_max=None) * \
            np.clip((z2 - z1), a_min=0, a_max=None)  # [N, M]
    return inter / (area1[:, None] + area2 - inter)


def box_size_np(boxes: ndarray) -> ndarray:
    """
    Compute length of boxes along all dimensions
    
    Args:
        boxes (ndarray): boxes (x1, y1, x2, y2, z1, z2)[N, dim * 2]
    
    Returns:
        ndarray: size along axis (x, y, (z))[N, dim]
    """
    dists = []
    dists.append(boxes[:, 2] - boxes[:, 0])
    dists.append(boxes[:, 3] - boxes[:, 1])
    if boxes.shape[1] // 2 == 3:
        dists.append(boxes[:, 5] - boxes[:, 4])
    return np.stack(dists, axis=-1)

def box_center_np(boxes: np.ndarray) -> np.ndarray:
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
    return np.stack(centers, axis=1)
