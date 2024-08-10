# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

"""
Direct copy of nnU-Net
All credits go to: https://github.com/MIC-DKFZ/nnUNet
"""


from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from torch.nn import functional as F

from nndet.preprocessing.resampling import get_do_separate_z, get_new_shape


def torch_resample_patient(
    data: Union[torch.Tensor, np.ndarray, None],
    seg: Union[torch.Tensor, np.ndarray, None],
    original_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    target_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    force_separate_z: Union[bool, None] = False,
    separate_z_anisotropy_threshold: float = 3,
):
    """
    Resample data and segmentation to new spacing

    Args:
        data: input data
        seg: input segmentation
        original_spacing: original spacing
        target_spacing: target spacing
        force_separate_z: force separate lowres axis as z-axis
        separate_z_anisotropy_threshold: anisotropy threshold for separating z-axis

    Returns:
        Union[torch.Tensor, np.ndarray, None]: resampled data
        Union[torch.Tensor, np.ndarray, None]: resampled segmentation
    """
    new_shape = get_new_shape(data, seg, original_spacing, target_spacing)
    do_separate_z, axis = get_do_separate_z(
        original_spacing, target_spacing, force_separate_z, separate_z_anisotropy_threshold
    )

    if data is not None:
        data_reshaped = resample_data_or_seg(data, new_shape, False, axis, do_separate_z)
    else:
        data_reshaped = None

    if seg is not None:
        seg_reshaped = resample_data_or_seg(seg, new_shape, True, axis, do_separate_z)
    else:
        seg_reshaped = None

    return data_reshaped, seg_reshaped


def torch_resampler(
    data: torch.Tensor,
    new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
    is_seg: bool = False,
    num_threads: int = 4,
    device: torch.device = torch.device("cpu"),
    memefficient_seg_resampling: bool = False,
    mode: str = "linear",
):
    """
    Resample the given data into a given shape

    Args:
        data: array to resample [C, dims]
        new_shape: define new dims (without channels)
        is_seg: changes the resampling strategy
        num_threads: number of threads to use for resampling
        device: device to be used for resampling
        memefficient_seg_resampling: execute slow but memory efficient resampling 
        mode: algorithm to be used for resampling. Available options are: nearest,
            linear (3D-only), bilinear, bicubic (4D-only), trilinear (5D-only),
            area, nearest-exact

        Returns:
            torch.Tensor: resampled array
    """
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

    if is_seg and torch_mode != "nearest":
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


def resample_data_or_seg(
    data: Union[torch.Tensor, np.ndarray],
    new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
    is_seg: bool = False,
    axis=None,
    do_separate_z=False,
    num_threads: int = 4,
    device: torch.device = torch.device("cpu"),
    memefficient_seg_resampling: bool = False,
    mode: str = "linear",
    aniso_axis_mode: str = "nearest-exact",
) -> np.ndarray:
    """
    Resample data or segmentation
    Direct copy of nnunet: https://github.com/MIC-DKFZ/nnUNet

    Args:
        data: array to resample [C, dims]
        new_shape: define new dims (without channels)
        is_seg: changes the resampling strategy
        axis: anisotropic axis, different resampling order used here
        do_separate_z: Different resampling along z dimensions
        num_threads: number of threads to use for resampling
        device: device to be used for resampling
        memefficient_seg_resampling: execute slow but memory efficient resampling 
        mode: algorithm to be used for resampling. Available options are: nearest,
            linear (3D-only), bilinear, bicubic (4D-only), trilinear (5D-only),
            area, nearest-exact
    Returns:
        np.ndarray: resampled array
    """
    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == len(data.shape) - 1

    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    
    was_numpy = isinstance(data, np.ndarray)
    if was_numpy:
        data = torch.from_numpy(data)
    else:
        orig_device = deepcopy(data.device)
        
    if np.any(shape != new_shape):
        if do_separate_z:
            print("separate z")
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
            data = torch_resampler(data, tmp_new_shape, is_seg, num_threads, device, memefficient_seg_resampling, mode)
            data = rearrange(
                data,
                f"(c {axis_letter}) {others[0]} {others[1]} -> c x y z",
                **{axis_letter: shape[axis], others[0]: tmp_new_shape[0], others[1]: tmp_new_shape[1]},
            )
            # reshape out of plane w/ nearest
            data = torch_resampler(data, new_shape, is_seg, num_threads, device, memefficient_seg_resampling, aniso_axis_mode)
        else:
            print("no separate z")
            data = torch_resampler(data, new_shape, is_seg, num_threads, device, memefficient_seg_resampling)

    else:
        print("no resampling necessary")
        
    if was_numpy:
        data = data.cpu().numpy().astype(dtype_data)
    else:
        data = data.to(orig_device).type(dtype_data)
    return data
