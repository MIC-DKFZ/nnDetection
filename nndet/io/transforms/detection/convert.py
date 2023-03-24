# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Sequence

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform


class Convert3DTo2DTransform(AbstractTransform):
    def __init__(self, array_keys: Sequence[str], point_key: Optional[str] = None):
        """
        Utility transform to overload the channel dimension with additional
        spatial information. The resulting data will be 2D converted from 3D.

        Args:
            array_keys: keys to covnert from 3d to 2d. Shapes will be cached in
                ""orig_shape_{k}" so these keys should be empty.
            point_key: key where points are located.
        """
        self.array_keys = array_keys
        self.point_key = point_key

    def __call__(self, **data_dict):
        return convert_3d_to_2d_generator(
            data_dict,
            array_keys=self.array_keys,
            point_key=self.point_key,
        )


class Convert2DTo3DTransform(AbstractTransform):
    def __init__(self, array_keys: Sequence[str], point_key: Optional[str] = None):
        """
        Utility transform to restore the spatial dimensions from previously
        overloading the channel dimension. The resulting data will be 3D
        converted from 2D.

        Args:
            array_keys: keys to convert from 2d to 3d. Shapes will be cached in
                ""orig_shape_{k}" so these keys should be empty.
            point_key: key where points are located.
        """
        self.array_keys = array_keys
        self.point_key = point_key

    def __call__(self, **data_dict):
        return convert_2d_to_3d_generator(
            data_dict,
            array_keys=self.array_keys,
            point_key=self.point_key,
        )


def convert_3d_to_2d_generator(
    data_dict: dict,
    array_keys: Sequence[str],
    point_key: Optional[str],
) -> dict:
    for k in array_keys:
        shp = data_dict[k].shape
        data_dict[k] = data_dict[k].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
        data_dict[f"orig_shape_{k}"] = shp
    if point_key is not None:
        data_dict[f"orig_coords_{point_key}"] = [p[..., 2] for p in data_dict[point_key]]
        data_dict[point_key] = [p[..., [0, 1, 3]] for p in data_dict[point_key]]
    return data_dict


def convert_2d_to_3d_generator(
    data_dict: dict,
    array_keys: Sequence[str],
    point_key: Optional[str],
) -> dict:
    for k in array_keys:
        shp = data_dict[f"orig_shape_{k}"]
        current_shape = data_dict[k].shape
        data_dict[k] = data_dict[k].reshape((shp[0], shp[1], shp[2], current_shape[-2], current_shape[-1]))
    if point_key is not None:
        data_dict[point_key] = [
            np.insert(
                p,
                obj=2,
                values=p_insert,
                axis=-1,
            )
            for p, p_insert in zip(data_dict[point_key], data_dict[f"orig_coords_{point_key}"])
        ]
        data_dict.pop(f"orig_coords_{point_key}")
    return data_dict
