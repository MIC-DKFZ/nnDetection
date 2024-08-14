# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

"""
Direct copy of nnU-Net
All credits go to: https://github.com/MIC-DKFZ/nnUNet
"""

from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from batchgenerators.augmentations.utils import resize_segmentation
from scipy.ndimage.interpolation import map_coordinates
from skimage.transform import resize


def get_lowres_axis(new_spacing):
    """
    Direct copy of nnunet: https://github.com/MIC-DKFZ/nnUNet
    """
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis


def get_do_separate_z(
    original_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    target_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    force_separate_z: Union[bool, None],
    separate_z_anisotropy_threshold: float,
) -> Union[bool, np.ndarray]:
    """
    Determine whether or not to do separate z resampling and along which axis

    Args:
        original_spacing: original spacing
        target_spacing: target spacing
        force_separate_z: force separate lowres axis as z-axis
        separate_z_anisotropy_threshold: anisotropy threshold for separating z-axis

        Returns:
            bool: whether or not to do separate z resampling
            Union[np.ndarray | None]: anisotropic axis
    """
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(original_spacing)
        else:
            axis = None
    else:
        if (np.max(original_spacing) / np.min(original_spacing)) > separate_z_anisotropy_threshold:
            do_separate_z = True
            axis = get_lowres_axis(original_spacing)
        elif (np.max(target_spacing) / np.min(target_spacing)) > separate_z_anisotropy_threshold:
            do_separate_z = True
            axis = get_lowres_axis(target_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            # every axis has the spacing, this should never happen, why is this code here?
            do_separate_z = False
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
        else:
            pass
    return do_separate_z, axis


def get_new_shape(
    data: Union[torch.Tensor, np.ndarray, None],
    seg: Union[torch.Tensor, np.ndarray, None],
    original_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    target_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
) -> np.ndarray:
    """
    Determine the shape of the resampled array

    Args:
        data: input data
        seg: input segmentation
        original_spacing: original spacing
        target_spacing: target spacing

    Returns:
        np.ndarray: new shape of the resampled array
    """
    assert not ((data is None) and (seg is None))
    if data is not None:
        assert len(data.shape) == 4, "data must be c x y z"
    if seg is not None:
        assert len(seg.shape) == 4, "seg must be c x y z"

    if data is not None:
        shape = np.array(data[0].shape)
    else:
        shape = np.array(seg[0].shape)

    new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(int)

    return new_shape


def resample_patient(
    data: np.ndarray,
    seg: Optional[np.ndarray],
    original_spacing: Sequence[float],
    target_spacing: Sequence[float],
    order_data: int = 3,
    order_seg: int = 0,
    force_separate_z: bool = False,
    order_z_data: int = 0,
    order_z_seg: int = 0,
    separate_z_anisotropy_threshold: float = 3,
) -> Union[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Direct copy of nnunet: https://github.com/MIC-DKFZ/nnUNet
    """
    new_shape = get_new_shape(
        data=data,
        seg=seg,
        original_spacing=original_spacing,
        target_spacing=target_spacing,
    )
    do_separate_z, axis = get_do_separate_z(
        original_spacing=original_spacing,
        target_spacing=target_spacing,
        force_separate_z=force_separate_z,
        separate_z_anisotropy_threshold=separate_z_anisotropy_threshold,
    )

    if data is not None:
        data_reshaped = resample_data_or_seg(
            data=data,
            new_shape=new_shape,
            is_seg=False,
            axis=axis,
            order=order_data,
            do_separate_z=do_separate_z,
            order_z=order_z_data,
        )
    else:
        data_reshaped = None

    if seg is not None:
        seg_reshaped = resample_data_or_seg(
            data=seg,
            new_shape=new_shape,
            is_seg=True,
            axis=axis,
            order=order_seg,
            do_separate_z=do_separate_z,
            order_z=order_z_seg,
        )
    else:
        seg_reshaped = None

    return data_reshaped, seg_reshaped


def resample_data_or_seg(
    data: np.ndarray,
    new_shape: Sequence[int],
    is_seg: bool,
    axis: Optional[Sequence[int]] = None,
    order: int = 3,
    do_separate_z: bool = False,
    order_z: int = 0,
) -> np.ndarray:
    """
    Resample data or segmentation
    Direct copy of nnunet: https://github.com/MIC-DKFZ/nnUNet

    Args:
        data: array to resample [C, dims]
        new_shape: define new dims (without channels)
        is_seg: changes the resampling strategy
        axis: anisotropic axis, different resampling order used here
        order: order of resampling along the isotropic axis
        do_separate_z: Different resampling along z dimensions
        order_z: if separate z resampling is done then this is the order for resampling in z

    Returns:
        np.ndarray: resampled array
    """
    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == len(data.shape) - 1
    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {"mode": "edge", "anti_aliasing": False}
    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)

    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_separate_z:
            print("separate z, order in z is", order_z, "order inplane is", order)
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []
            for c in range(data.shape[0]):
                reshaped_data = []
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_data.append(
                            resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data)
                        )
                    elif axis == 1:
                        reshaped_data.append(
                            resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data)
                        )
                    else:
                        reshaped_data.append(
                            resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data)
                        )
                reshaped_data = np.stack(reshaped_data, axis)
                if shape[axis] != new_shape[axis]:

                    # The following few lines are blatantly copied and modified from sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final_data.append(
                            map_coordinates(reshaped_data, coord_map, order=order_z, mode="nearest")[None].astype(
                                dtype_data
                            )
                        )
                    else:
                        unique_labels = np.unique(reshaped_data)
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        for i, cl in enumerate(unique_labels):
                            reshaped_multihot = np.round(
                                map_coordinates(
                                    (reshaped_data == cl).astype(float),
                                    coord_map,
                                    order=order_z,
                                    mode="nearest",
                                )
                            )
                            reshaped[reshaped_multihot > 0.5] = cl
                        reshaped_final_data.append(reshaped[None].astype(dtype_data))
                else:
                    reshaped_final_data.append(reshaped_data[None].astype(dtype_data))
            reshaped_final_data = np.vstack(reshaped_final_data)
        else:
            print("no separate z, order", order)
            reshaped = []
            for c in range(data.shape[0]):
                reshaped.append(resize_fn(data[c], new_shape, order, **kwargs)[None].astype(dtype_data))
            reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        print("no resampling necessary")
        return data
