from typing import Dict, Hashable, List, Optional, Sequence

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform


class MirrorTransform(AbstractTransform):
    def __init__(
        self,
        data_key: Hashable,
        label_key: Optional[Hashable] = None,
        point_key: Optional[Hashable] = None,
        p_per_sample: float = 1,
        axes: Sequence[int] = (0, 1, 2),
    ):
        """
        Randomly mirrors data, seg and points along specified axes.
        Mirroring is evenly distributed. Probability of mirroring along each
        axis is `0.5` .
        This function is adapted from batchgenerators.

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
            axes: Specifies the axes to mirror. Defaults to (0, 1, 2).

        Raises:
            ValueError: raised if maximum of axes exceeds 2.
        """
        self.data_key = data_key
        self.label_key = label_key
        self.point_key = point_key

        self.axes = axes
        self.p_per_sample = p_per_sample

        if max(axes) > 2:
            raise ValueError(
                "MirrorTransform now takes the axes as the spatial dimensions. What previously was "
                "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                "is now axes=(0, 1, 2). Please adapt your scripts accordingly."
            )

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
                mirror_axes = self.get_axes(axes=self.axes)
                if points is not None:
                    mirror_matrix = self.get_matrix(axes=mirror_axes, img_shape=img_shape)

                if mirror_axes:
                    data[b] = mirror_array(data[b], axes=mirror_axes)
                    if seg is not None:
                        seg[b] = mirror_array(seg[b], axes=mirror_axes)
                    if points is not None:
                        points[b] = mirror_points(points[b], matrix=mirror_matrix)

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
        Retrieve axes to mirror. Each axis is mirrored with probability 0.5.

        Args:
            axes: axes to mirror

        Returns:
            List[int]: selected axes
        """
        selecetd_axes = []
        if 0 in axes and np.random.uniform() < 0.5:
            selecetd_axes.append(0)
        if 1 in axes and np.random.uniform() < 0.5:
            selecetd_axes.append(1)
        if 2 in axes and np.random.uniform() < 0.5:
            selecetd_axes.append(2)
        return selecetd_axes

    @staticmethod
    def get_matrix(
        axes: Sequence[int],
        img_shape: List[int],
    ) -> np.ndarray:
        """
        Create matrix to mirror points

        Args:
            axes: axis to mirror
            img_shape: shape of image

        Returns:
            Tensor: matrix for mirroring in homogeneous coordinates
        """
        axes = tuple(axes)
        cartesian_dims = len(img_shape)

        homogeneous_dims = cartesian_dims + 1
        mat = np.eye(homogeneous_dims, dtype=float)

        # reflection
        for a in axes:
            mat[a, a] = -1

        # add data shape to axis which were reflected
        self_tensor = np.zeros(cartesian_dims, dtype=float)
        index_tensor = np.array(axes, dtype=int)
        src_tensor = np.array([1] * len(axes), dtype=float)
        np.put_along_axis(self_tensor, index_tensor, src_tensor, 0)
        mat[:-1, -1] = self_tensor * (np.array(img_shape) - 1)
        return mat


def mirror_array(
    data: np.ndarray,
    axes: Sequence[int],
) -> np.ndarray:
    """
    Mirror array along provided axis

    Args:
        data: input array [C, dims], where is the color channel and
            dims are sptatial dimensions
        axes: axis to mirror, 0 refers to the first spatial dimension
            not the color channel!

    Returns:
        np.ndarray: mirrored array
    """
    _axes = [a + 1 for a in axes]
    return np.flip(data, tuple(_axes))


def mirror_points(
    points: np.ndarray,
    matrix: Sequence[int],
) -> np.ndarray:
    """
    Mirror points with mirror matrix

    Args:
        points: points to mirror. Expects points to be in the format
            [R, L, dims + 1] where  R is the number of objects,
            L is the number of points per object and dims are the number
            of spatial dimensions
        matrix: mirror matrix

    Returns:
        np.ndarray: mirrored points (same format as input points)
    """
    if points.size == 0:
        return points
    return points @ matrix.T
