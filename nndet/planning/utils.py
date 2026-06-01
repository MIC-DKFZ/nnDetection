# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

"""
Direct copy of nnU-Net, all credits to https://github.com/MIC-DKFZ/nnUNet
"""

from copy import deepcopy

import numpy as np


def get_pool_and_conv_props(spacing, patch_size, min_feature_map_size, max_numpool):
    """
    Direct copy of nnU-Net, all credits to https://github.com/MIC-DKFZ/nnUNet

    :param spacing:
    :param patch_size:
    :param min_feature_map_size: min edge length of feature maps in bottleneck
    :return:
    """
    dim = len(spacing)

    current_spacing = deepcopy(list(spacing))
    current_size = deepcopy(list(patch_size))

    pool_op_kernel_sizes = []
    conv_kernel_sizes = []

    num_pool_per_axis = [0] * dim

    while True:
        # This is a problem because sometimes we have spacing 20, 50, 50 and we want to still keep pooling.
        # Here we would stop however. This is not what we want! Fixed in get_pool_and_conv_propsv2
        min_spacing = min(current_spacing)
        valid_axes_for_pool = [i for i in range(dim) if current_spacing[i] / min_spacing < 2]
        axes = []
        for a in range(dim):
            my_spacing = current_spacing[a]
            partners = [
                i for i in range(dim) if current_spacing[i] / my_spacing < 2 and my_spacing / current_spacing[i] < 2
            ]
            if len(partners) > len(axes):
                axes = partners
        conv_kernel_size = [3 if i in axes else 1 for i in range(dim)]

        # exclude axes that we cannot pool further because of min_feature_map_size constraint
        # before = len(valid_axes_for_pool)
        valid_axes_for_pool = [i for i in valid_axes_for_pool if current_size[i] >= 2 * min_feature_map_size]
        # after = len(valid_axes_for_pool)
        # if after == 1 and before > 1:
        #    break

        valid_axes_for_pool = [i for i in valid_axes_for_pool if num_pool_per_axis[i] < max_numpool]

        if len(valid_axes_for_pool) == 0:
            break

        # print(current_spacing, current_size)

        other_axes = [i for i in range(dim) if i not in valid_axes_for_pool]

        pool_kernel_sizes = [0] * dim
        for v in valid_axes_for_pool:
            pool_kernel_sizes[v] = 2
            num_pool_per_axis[v] += 1
            current_spacing[v] *= 2
            current_size[v] = np.ceil(current_size[v] / 2)
        for nv in other_axes:
            pool_kernel_sizes[nv] = 1

        pool_op_kernel_sizes.append(pool_kernel_sizes)
        conv_kernel_sizes.append(conv_kernel_size)
        # print(conv_kernel_sizes)

    must_be_divisible_by = get_shape_must_be_divisible_by(num_pool_per_axis)
    patch_size = pad_shape(patch_size, must_be_divisible_by)

    # we need to add one more conv_kernel_size for the bottleneck. We always use 3x3(x3) conv here
    conv_kernel_sizes.append([3] * dim)
    return num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, must_be_divisible_by


def get_shape_must_be_divisible_by(net_numpool_per_axis):
    """
    Direct copy of nnU-Net, all credits to https://github.com/MIC-DKFZ/nnUNet
    """
    return 2 ** np.array(net_numpool_per_axis)


def pad_shape(shape, must_be_divisible_by):
    """
    Direct copy of nnU-Net, all credits to https://github.com/MIC-DKFZ/nnUNet

    pads shape so that it is divisibly by must_be_divisible_by
    :param shape:
    :param must_be_divisible_by:
    :return:
    """
    if not isinstance(must_be_divisible_by, (tuple, list, np.ndarray)):
        must_be_divisible_by = [must_be_divisible_by] * len(shape)
    else:
        assert len(must_be_divisible_by) == len(shape)

    new_shp = [shape[i] + must_be_divisible_by[i] - shape[i] % must_be_divisible_by[i] for i in range(len(shape))]

    for i in range(len(shape)):
        if shape[i] % must_be_divisible_by[i] == 0:
            new_shp[i] -= must_be_divisible_by[i]
    new_shp = np.array(new_shp).astype(int)
    return new_shp
