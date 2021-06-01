from typing import Sequence, List, Union, Tuple

import torch
import numpy as np
from torch import Tensor

from nndet.core.boxes import box_center


def scale_with_abs_strides(seq: Sequence[float],
                           strides: Sequence[Union[Sequence[Union[int, float]], Union[int, float]]],
                           dim_idx: int,
                           ) -> List[Tuple[float]]:
    """
    Scale values with absolute stride between feature maps
    
    Args:
        seq: sequence to scale
        strides: strides to scale with.
        dim_idx: dimension index for stride
    """
    scaled = []
    for stride in strides:
        if not isinstance(stride, (float, int)):
            _stride = stride[dim_idx]
        else:
            _stride = stride
        _scaled = [i * _stride for i in seq]
        scaled.append(tuple(_scaled))
    return scaled


def proxy_num_boxes_in_patch(boxes: Tensor, patch_size: Sequence[int]) -> Tensor:
    """
    This is just a proxy and not the exact computation

    Args:
        boxes: boxes
        patch_size: patch size

    Returns:
        Tensor: count of boxes which center point is in the range of patch_size / 2
    """
    patch_size = torch.tensor(patch_size, dtype=torch.float)[None, None] / 2 # [1, 1, dims]    

    center = box_center(boxes)  # [N, dims]
    center_dists = (center[None] - center[:, None]).abs()  # [N, N, dims]

    center_in_range = (center_dists <= patch_size).prod(dim=-1)  # [N, N]
    return center_in_range.sum(dim=1)  # [N]


def comp_num_pool_per_axis(patch_size: Sequence[int],
                           max_num_pool: int,
                           min_feature_map_size: int) -> List[int]:
    """
    Computes the maximum number of pooling operations given a minimal feature map size
    and the patch size

    Args:
        patch_size: input patch size
        max_num_pool: maximum number of pooling operations.
        min_feature_map_size: Minimal size of feature map inside the bottleneck.

    Returns:
        List[int]: max number of pooling operations per axis
    """
    network_numpool_per_axis = np.floor(
        [np.log(i / min_feature_map_size) / np.log(2) for i in patch_size]).astype(np.int32)
    network_numpool_per_axis = [min(i, max_num_pool) for i in network_numpool_per_axis]
    return network_numpool_per_axis


def get_shape_must_be_divisible_by(num_pool_per_axis: Sequence[int]) -> np.ndarray:
    """
    Returns a multiple of 2 which indicates by which factor an axis needs to
    be dividable to avoid problems with upsampling
    
    Args:
        num_pool_per_axis: number of pooling operations per axis
    
    Returns:
        np.ndarray: necessary divisor of axis
    """
    return 2 ** np.array(num_pool_per_axis)


def pad_shape(shape: Sequence[int], must_be_divisible_by: Sequence[int]) -> np.ndarray:
    """
    Pads shape so that it is divisibly by must_be_divisible_by
    
    Args:
        shape: shape to pad
        must_be_divisible_by: divisor
    
    Returns:
        np.ndarray: padded shape
    """
    if not isinstance(must_be_divisible_by, (tuple, list, np.ndarray)):
        must_be_divisible_by = [must_be_divisible_by] * len(shape)
    else:
        assert len(must_be_divisible_by) == len(shape)

    new_shp = [shape[i] + must_be_divisible_by[i] - shape[i] % must_be_divisible_by[i]
               for i in range(len(shape))]

    for i in range(len(shape)):
        if shape[i] % must_be_divisible_by[i] == 0:
            new_shp[i] -= must_be_divisible_by[i]
    new_shp = np.array(new_shp).astype(np.int32)
    return new_shp


def scale_with_abs_strides(seq: Sequence[float],
                           strides: Sequence[Union[Sequence[Union[int, float]], Union[int, float]]],
                           dim_idx: int,
                           ) -> List[Tuple[float]]:
    """
    Scale values with absolute stride between feature maps
    
    Args:
        seq: sequence to scale
        strides: strides to scale with.
        dim_idx: dimension index for stride
    """
    scaled = []
    for stride in strides:
        if not isinstance(stride, (float, int)):
            _stride = stride[dim_idx]
        else:
            _stride = stride
        _scaled = [i * _stride for i in seq]
        scaled.append(tuple(_scaled))
    return scaled


def proxy_num_boxes_in_patch(boxes: Tensor, patch_size: Sequence[int]) -> Tensor:
    """
    This is just a proxy and not the exact computation

    Args:
        boxes: boxes
        patch_size: patch size

    Returns:
        Tensor: count of boxes which center point is in the range of patch_size / 2
    """
    patch_size = torch.tensor(patch_size, dtype=torch.float)[None, None] / 2 # [1, 1, dims]    

    center = box_center(boxes)  # [N, dims]
    center_dists = (center[None] - center[:, None]).abs()  # [N, N, dims]

    center_in_range = (center_dists <= patch_size).prod(dim=-1)  # [N, N]
    return center_in_range.sum(dim=1)  # [N]


def fixed_anchor_init(dim: int):
    """
    Fixed anchors sizes for 2d and 3d

    Args:
        dim: number of dimensions

    Returns:
        dict: fixed params
    """
    anchor_plan = {"stride": 1, "aspect_ratios": (0.5, 1, 2)}
    if dim == 2:
        anchor_plan["sizes"] = (32, 64, 128, 256)
    else:
        anchor_plan["sizes"] = ((4, 8, 16), (8, 16, 32), (16, 32, 64), (32, 64, 128))
        anchor_plan["zsizes"] = ((2, 3, 4), (4, 6, 8), (8, 12, 16), (12, 24, 48))
    return anchor_plan


def concatenate_property_boxes(all_boxes: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([b for b in all_boxes if not isinstance(b, list) and b.size > 0], axis=0)
