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

from typing import Sequence, Tuple, Union, Optional

import numpy as np
from loguru import logger

from nndet.core.boxes.ops import permute_boxes, expand_to_boxes
from nndet.preprocessing.resampling import resample_data_or_seg, get_do_separate_z, get_lowres_axis


def restore_detection(boxes: np.ndarray,
                      transpose_backward: Sequence[int],
                      original_spacing: Sequence[float],
                      spacing_after_resampling: Sequence[float],
                      crop_bbox: Sequence[Tuple[int, int]],
                      **kwargs,
                      ) -> np.ndarray:
    """
    Restore boxes from preprocessed space into original space

    Args:
        boxes: predicted boxes in preprocessing space,
            (x1, y1, x2, y2, (z1, z2))[N, dims * 2]
        transpose_backward: backward transposing
        original_spacing: spacing of the original image
        spacing_after_resampling: spacing in the preprocessed space
            (forward transposed order)
        crop_bbox: bounding box crop in the original spacing [(min_0, max_0), ...]
        **kwargs: ignored

    Returns:
        np.ndarray: predicted bounding boxes in the original image space
    """
    boxes_transposed = permute_boxes(boxes, transpose_backward)

    original_spacing = np.asarray(original_spacing)
    spacing_after_resampling = np.asarray(spacing_after_resampling)
    resampled_spacing = spacing_after_resampling[transpose_backward]
    scaling = resampled_spacing / original_spacing
    scaling_expanded = expand_to_boxes(scaling[None])
    boxes_scaled = boxes_transposed * scaling_expanded

    offset = np.asarray([i[0] for i in crop_bbox])
    offset_expanded = expand_to_boxes(offset[None])
    boxes_original = boxes_scaled + offset_expanded
    return boxes_original


def restore_fmap(fmap: np.ndarray,
                 transpose_backward: Sequence[int],
                 original_spacing: Sequence[float],
                 spacing_after_resampling: Sequence[float],
                 original_size_before_cropping: Sequence[int],
                 size_after_cropping: Sequence[int],
                 crop_bbox: Optional[Sequence[Tuple[int, int]]] = None,
                 interpolation_order: int = 3,
                 interpolation_order_z: int = 0,
                 do_separate_z: bool = None,
                 ) -> np.ndarray:
    """
    Restore feature map from preprocessed space into original space

    Args:
        fmap: feature map to resample [C, dims], where C is the number of
            channels
        transpose_backward: backward transposing
        original_spacing: spacing of the original image
        spacing_after_resampling: spacing in the preprocessed space
            (forward transposed order)
        original_size_before_cropping: original image size before cropping
        size_after_cropping: image size after cropping
        crop_bbox: bounding box of crop
        interpolation_order: interpolation order for inplane axis
        interpolation_order_z: interpolation order for anisotropic axis
        do_separate_z: if None then we dynamically decide how to resample
            along z, if True/False then always/never resample along z
            separately. Do not touch unless you know what you are doin

    Returns:
        np.ndarray: resampled feature map [C, new_dims]
    """
    fmap_transposed = np.transpose(fmap, [0] + [i + 1 for i in transpose_backward])

    original_spacing = np.asarray(original_spacing)
    spacing_after_resampling = np.asarray(spacing_after_resampling)
    resampled_spacing = spacing_after_resampling[transpose_backward]

    if np.any([i != j for i, j in zip(fmap_transposed.shape[1:], size_after_cropping)]):
        lowres_axis = _get_lowres_axes(original_spacing, resampled_spacing,
                                       do_separate_z=do_separate_z)
        logger.info(f"Resampling: do separate z: {do_separate_z}; lowres axis: {lowres_axis}")
        fmap_old_spacing = resample_data_or_seg(fmap_transposed, size_after_cropping, is_seg=False,
                                                axis=lowres_axis, order=interpolation_order,
                                                do_separate_z=do_separate_z,
                                                order_z=interpolation_order_z)
    else:
        logger.info(f"Resampling: no resampling necessary")
        fmap_old_spacing = fmap_transposed

    if crop_bbox is not None:
        crop_bbox = [list(cb) for cb in crop_bbox]
        tmp = np.zeros((fmap_old_spacing.shape[0], *original_size_before_cropping))

        for c in range(len(crop_bbox)):
            crop_bbox[c][1] = np.min(
                (crop_bbox[c][0] + fmap_old_spacing.shape[c + 1], original_size_before_cropping[c]))

        _slices = [...] + [slice(b[0], b[1]) for b in crop_bbox]
        tmp[_slices] = fmap_old_spacing
        fmap_original = tmp
    else:
        fmap_original = fmap_old_spacing
    return fmap_original


def _get_lowres_axes(original_spacing: Sequence[float],
                     resampled_spacing: Sequence[float],
                     do_separate_z: bool) -> Union[Sequence[int], None]:
    """
    Dynamically determine lowres axes

    Args:
        original_spacing: original spacing (not transposed!)
        resampled_spacing: resampled sapcing (not transposed!)
        do_separate_z: force sepearte

    Returns:
        Union[Sequence[int], None]: Lowres axes. If None, no lowres axes
            is present
    """
    if do_separate_z is None:
        if get_do_separate_z(original_spacing):  # original spacing was anisotropic
            do_separate_z = True
            lowres_axis = get_lowres_axis(original_spacing)
        elif get_do_separate_z(resampled_spacing):  # resampled spacing was anisotropic
            do_separate_z = True
            lowres_axis = get_lowres_axis(resampled_spacing)
        else:  # no separate z
            do_separate_z = False
            lowres_axis = None
    else:
        if do_separate_z:
            lowres_axis = get_lowres_axis(original_spacing)
        else:
            lowres_axis = None
    return lowres_axis
