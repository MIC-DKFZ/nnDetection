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

from nndet.utils.info import SuppressPrint

with SuppressPrint():
    import nnunet.preprocessing.preprocessing as nn_preprocessing


def resize_segmentation(segmentation, new_shape, order=3):
    """
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    """
    return nn_preprocessing.resize_segmentation(
        segmentation=segmentation, new_shape=new_shape, order=order)


def get_do_separate_z(spacing, anisotropy_threshold: float = 3):
    return nn_preprocessing.get_do_separate_z(spacing=spacing, anisotropy_threshold=anisotropy_threshold)


def get_lowres_axis(new_spacing):
    return nn_preprocessing.get_lowres_axis(new_spacing=new_spacing)


def resample_patient(data,
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
    return nn_preprocessing.resample_patient(data=data, seg=seg, original_spacing=original_spacing,
                                             target_spacing=target_spacing, order_data=order_data,
                                             order_seg=order_seg, force_separate_z=force_separate_z,
                                             order_z_data=order_z_data,
                                             order_z_seg=order_z_seg,
                                             separate_z_anisotropy_threshold=separate_z_anisotropy_threshold)


def resample_data_or_seg(data, new_shape, is_seg, axis=None, order=3,
                         do_separate_z=False, order_z=0) -> np.ndarray:
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
        data=data, new_shape=new_shape, is_seg=is_seg, axis=axis,
        order=order, do_separate_z=do_separate_z, order_z=order_z)
