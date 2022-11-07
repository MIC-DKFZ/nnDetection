# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from nndet.utils.info import SuppressPrint

with SuppressPrint():
    import nnunet.preprocessing.preprocessing as nn_preprocessing


def get_do_separate_z(spacing, anisotropy_threshold: float = 3):
    return nn_preprocessing.get_do_separate_z(spacing=spacing, anisotropy_threshold=anisotropy_threshold)


def get_lowres_axis(new_spacing):
    return nn_preprocessing.get_lowres_axis(new_spacing=new_spacing)


def resample_patient(
    data,
    seg,
    original_spacing,
    target_spacing,
    order_data=3,
    order_seg=0,
    force_separate_z=False,
    order_z_data=0,
    order_z_seg=0,
    separate_z_anisotropy_threshold: float = 3,
):
    return nn_preprocessing.resample_patient(
        data=data,
        seg=seg,
        original_spacing=original_spacing,
        target_spacing=target_spacing,
        order_data=order_data,
        order_seg=order_seg,
        force_separate_z=force_separate_z,
        order_z_data=order_z_data,
        order_z_seg=order_z_seg,
        separate_z_anisotropy_threshold=separate_z_anisotropy_threshold,
    )


def resample_data_or_seg(data, new_shape, is_seg, axis=None, order=3, do_separate_z=False, order_z=0) -> np.ndarray:
    """
    Resample data or segmentation

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
    return nn_preprocessing.resample_data_or_seg(
        data=data,
        new_shape=new_shape,
        is_seg=is_seg,
        axis=axis,
        order=order,
        do_separate_z=do_separate_z,
        order_z=order_z,
    )
