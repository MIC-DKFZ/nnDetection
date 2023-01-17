from functools import lru_cache
from typing import Dict, Hashable, List, Optional, Sequence

import numpy as np
from batchgenerators.augmentations.utils import (
    create_matrix_rotation_2d,
    create_matrix_rotation_x_3d,
    create_matrix_rotation_y_3d,
    create_matrix_rotation_z_3d,
)
from batchgenerators.transforms.abstract_transforms import AbstractTransform


class Rot90Transform(AbstractTransform):
    def __init__(
        self,
        data_key: Hashable,
        label_key: Optional[Hashable] = None,
        point_key: Optional[Hashable] = None,
        p_per_sample: float = 1,
        axes: Sequence[int] = (0, 1, 2),
        num_rot: Sequence[int] = (1, 2, 3),
    ):
        """
        Randomly rotates the data by 90 degrees

        Args:
            data_key: specify where data is located in dict.
                Expects data to be of format [B, C, s_dims], where B is the batch
                size, C number of of channels and s_dims are up to three
                spatial dimensions. Defaults to "data".
            label_key: specify where seg is located in dict.
                Expects seg to be of format [B, C, s_dims], where B is the batch
                size, C number of of channels and s_dims are up to three
                spatial dimensions. Defaults to "data".
            point_key: specify where points are located in dict.
                Expects points to be in the format List([R, L, dims + 1]) where
                the List is the batch dimension, R is the number of objects,
                L is the number of points per object and dims are the number
                of spatial dimensions
            p_per_sample: Probability to apply any mirroring to a given sample.
                Defaults to 1.
            axes: around which axes will the rotation take place? two axes
                are chosen randomly from axes.. Defaults to (0, 1, 2).
            num_rot: rotate by 90 degrees how often? must be tuple -> num rot
                randomly chosen from that tuple
        """
        self.data_key = data_key
        self.label_key = label_key
        self.point_key = point_key

        self.axes = axes
        self.p_per_sample = p_per_sample
        self.num_rot = num_rot

        assert min(num_rot) >= 0

    def __call__(self, **data_dict: Dict) -> Dict:
        # retrieve batch
        data = data_dict[self.data_key]
        batch_size = len(data)
        img_shape = tuple(data.shape[2:])

        if self.label_key is not None:
            seg = data_dict[self.label_key]
        else:
            seg = None

        if self.point_key is not None:
            points = data_dict[self.point_key]
        else:
            points = None

        # apply
        for b in range(batch_size):
            if np.random.uniform() < self.p_per_sample:
                _num_rot = np.random.choice(self.num_rot)
                rot_axes = self.get_axes(axes=self.axes)

                if points is not None:
                    rot90_matrix = self.get_matrix(axes=rot_axes, num_rot=_num_rot, ndim=len(img_shape))

                data[b] = rot90_array(data[b], axes=rot_axes, num_rot=_num_rot)
                if seg is not None:
                    seg[b] = rot90_array(seg[b], axes=rot_axes, num_rot=_num_rot)
                if points is not None:
                    points[b] = rot90_points(points[b], matrix=rot90_matrix)

        # save batch
        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg
        if points is not None:
            data_dict[self.point_key] = points
        return data_dict

    @staticmethod
    def get_axes(axes: Sequence[int]) -> List[int]:
        """
        Retrieve axes to rate inbetween.

        Args:
            axes: axes to rotate

        Returns:
            List[int]: axes between which the rotation will be performed
        """
        axes = np.random.choice(axes, size=2, replace=False)
        axes.sort()
        return axes

    @lru_cache(maxsize=None)
    @staticmethod
    def get_matrix(axes: Sequence[int], num_rot: int, ndim: int) -> np.ndarray:
        """
        Retrieve matrix to rotate points

        Args:
            axes: axes where the rotation is performed
            num_rot: number of 90 degree rotations
            ndim: number of spatial dimensions

        Raises:
            RuntimeError: raised if axes combination is not recognized

        Returns:
            np.ndarray: matrix to rotate points
        """
        mat = np.zeros((ndim + 1, ndim + 1))
        angle = np.pi * num_rot
        _axes = tuple(axes)

        if ndim == 2:
            mat[:2, :2] = create_matrix_rotation_2d(angle=angle)
        else:
            if _axes == (0, 1):
                m = create_matrix_rotation_x_3d(angle=angle)
            elif _axes == (1, 0):
                m = create_matrix_rotation_x_3d(angle=-angle)
            elif _axes == (0, 2):
                m = create_matrix_rotation_y_3d(angle=angle)
            elif _axes == (2, 0):
                m = create_matrix_rotation_y_3d(angle=-angle)
            elif _axes == (1, 2):
                m = create_matrix_rotation_z_3d(angle=angle)
            elif _axes == (2, 1):
                m = create_matrix_rotation_z_3d(angle=-angle)
            else:
                raise RuntimeError
            mat[:3, :3] = m
        return mat


def rot90_array(data: np.ndarray, axes: Sequence[int], num_rot: int) -> np.ndarray:
    """
    Perofrm rot 90 on data with color channel

    Args:
        data: data to rotate
        axes: axes to rotate inbetween
        num_rot: number ot rotations

    Returns:
        np.ndarray: rotated array
    """
    _axes = [i + 1 for i in axes]
    return np.rot90(data, num_rot=num_rot, axes=_axes)


def rot90_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Rotate points

    Args:
        points: points to rotate. Expects points to be in the format
            [R, L, dims + 1] where  R is the number of objects,
            L is the number of points per object and dims are the number
            of spatial dimensions
        matrix: rotation matrix

    Returns:
        np.ndarray: rotated points (same format as input points)
    """
    if points.size == 0:
        return points
    return points @ matrix.T
