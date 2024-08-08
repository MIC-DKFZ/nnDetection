# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

"""
Direct copy of nnU-Net
All credits go to: https://github.com/MIC-DKFZ/nnUNet
"""

import numpy as np
import torch
from einops import rearrange
from torch.nn import functional as F


def get_do_separate_z(spacing, anisotropy_threshold: float = 3):
    """
    Direct copy of nnunet: https://github.com/MIC-DKFZ/nnUNet
    """
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing):
    """
    Direct copy of nnunet: https://github.com/MIC-DKFZ/nnUNet
    """
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis


def torch_resample_patient(
    data,
    seg,
    original_spacing,
    target_spacing,
    force_separate_z=False,
    order_z_data=0,
    order_z_seg=0,
    separate_z_anisotropy_threshold: float = 3,
):
    """
    Direct copy of nnunet: https://github.com/MIC-DKFZ/nnUNet
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

    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(original_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(original_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(original_spacing)
        elif get_do_separate_z(target_spacing, separate_z_anisotropy_threshold):
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

    if data is not None:
        data_reshaped = resample_data_or_seg(data, new_shape, False, axis, do_separate_z, order_z=order_z_data)
    else:
        data_reshaped = None
    if seg is not None:
        seg_reshaped = resample_data_or_seg(seg, new_shape, True, axis, do_separate_z, order_z=order_z_seg)
    else:
        seg_reshaped = None
    return data_reshaped, seg_reshaped


def torch_resampler(
    data,
    new_shape,
    is_seg=False,
    num_threads=4,
    device=torch.device("cpu"),
    memefficient_seg_resampling=False,
    mode="linear",
):
    if mode == "linear":
        if data.ndim == 4:
            torch_mode = "trilinear"
        elif data.ndim == 3:
            torch_mode = "bilinear"
        else:
            raise RuntimeError
    else:
        torch_mode = mode

    n_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)

    data = torch.from_numpy(data).to(device)
    new_shape = tuple(new_shape)

    if is_seg and torch_mode != 'nearest':
        unique_values = torch.unique(data)
        result_dtype = torch.int8 if max(unique_values) < 127 else torch.int16
        result = torch.zeros((data.shape[0], *new_shape), dtype=result_dtype, device=device)
        if not memefficient_seg_resampling:
            result_tmp = torch.zeros(
                (len(unique_values), data.shape[0], *new_shape), dtype=torch.float16, device=device
            )
            scale_factor = 1000
            done_mask = torch.zeros_like(result, dtype=torch.bool, device=device)
            for i, u in enumerate(unique_values):
                result_tmp[i] = F.interpolate(
                    (data[None] == u).float() * scale_factor, new_shape, mode=torch_mode, antialias=False
                )[0]
                mask = result_tmp[i] > (0.7 * scale_factor)
                result[mask] = u.item()
                done_mask |= mask
            if not torch.all(done_mask):
                result[~done_mask] = unique_values[result_tmp[:, ~done_mask].argmax(0)].to(result_dtype)
        else:
            for i, u in enumerate(unique_values):
                if u == 0:
                    pass
                result[
                    F.interpolate((data[None] == u).float(), new_shape, mode=torch_mode, antialias=False)[0] > 0.5
                ] = u
    else:
        result = F.interpolate(data[None].float(), new_shape, mode=torch_mode, antialias=False)[0]

    torch.set_num_threads(n_threads)
    return result.cpu().numpy()


def resample_data_or_seg(data, new_shape, is_seg, axis=None, do_separate_z=False, order_z=0) -> np.ndarray:
    """
    Resample data or segmentation
    Direct copy of nnunet: https://github.com/MIC-DKFZ/nnUNet

    Args:
        data: array to resample [C, dims]
        new_shape: define new dims (without channels)
        is_seg: changes the resampling strategy
        axis: anisotropic axis, different resampling order used here
        do_separate_z: Different resampling along z dimensions
        order_z: if separate z resampling is done then this is the order for resampling in z

    Returns:
        np.ndarray: resampled array
    """
    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == len(data.shape) - 1

    resize_fn = torch_resampler
    kwargs = dict(
        is_seg=is_seg, num_threads=4, device=torch.device("cpu"), memefficient_seg_resampling=False, mode="linear"
    )

    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_separate_z:
            print("separate z, order in z is", order_z)
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            tmp = "xyz"
            axis_letter = tmp[axis]
            others_int = [i for i in range(3) if i != axis]
            others = [tmp[i] for i in others_int]

            # reshape by overloading c channel
            data = rearrange(data, f"c x y z -> (c {axis_letter}) {others[0]} {others[1]}")

            # reshape in-plane
            tmp_new_shape = [new_shape[i] for i in others_int]
            data = resize_fn(data, tmp_new_shape, **kwargs)
            data = rearrange(
                data,
                f"(c {axis_letter}) {others[0]} {others[1]} -> c x y z",
                **{axis_letter: shape[axis], others[0]: tmp_new_shape[0], others[1]: tmp_new_shape[1]},
            )
            # reshape out of plane w/ nearest
            data = resize_fn(data, new_shape, **kwargs).astype(dtype_data)
        else:
            print("no separate z")
            data = resize_fn(data, new_shape, **kwargs).astype(dtype_data)
    else:
        print("no resampling necessary")
    return data
