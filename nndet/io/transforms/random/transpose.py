from typing import Dict, Hashable, List, Optional, Sequence

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform


class TransposeAxesTransform(AbstractTransform):
    def __init__(
        self,
        data_key: Hashable,
        label_key: Optional[Hashable] = None,
        point_key: Optional[Hashable] = None,
        p_per_sample: float = 1,
        axes: Sequence[int] = (0, 1, 2),
    ):
        """
        This transform will randomly shuffle the axes of 'axes'.
        Requires your patch size to have the same dimension in all axes
        specified in `axes`. So if `axes=(0, 1, 2)` the shape must
        be `(128x128x128)` and cannotbe, for example `(128x128x96)`
        (`transpose_any_of_these=(0, 1)` would be the correct one here)!
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
            p_per_sample: Probability to apply any transposing to a given
                sample. Defaults to 1.
            axes: Specifies the axes to transpose. Defaults to (0, 1, 2).

        Raises:
            ValueError: raised if maximum of axes exceeds 2.
        """
        self.data_key = data_key
        self.label_key = label_key
        self.point_key = point_key

        self.axes = axes
        self.p_per_sample = p_per_sample

        # checks
        if max(self.axes) > 2:
            raise ValueError(
                "TransposeAxesTransform now takes the axes as the spatial dimensions. What previously was "
                "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                "is now axes=(0, 1, 2). Please adapt your scripts accordingly."
            )
        assert isinstance(self.axes, (list, tuple)), "transpose_any_of_these must be either list or tuple"
        assert len(self.axes) >= 2, (
            "len(transpose_any_of_these) must be >=2 -> we need at least 2 axes we " "can transpose"
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
                transpose_axes = self.get_axes(axes=self.axes, ndim=len(img_shape))
                if points is not None:
                    transpose_matrix = self.get_matrix(transpose_axes)

                data[b] = transpose_array(data[b], axes=transpose_axes)
                if seg is not None:
                    seg[b] = transpose_array(seg[b], axes=transpose_axes)
                if points is not None:
                    points[b] = transpose_array(points[b], matrix=transpose_matrix)

        # save batch
        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg
        if points is not None:
            data_dict[self.point_key] = points
        return data_dict

    @staticmethod
    def get_axes(axes: Sequence[int], ndim: int) -> List[int]:
        """
        Retrieve trasposed axes order. The result can be passed to the
        transpose function of numpy arrays.

        Args:
            axes: axes to transpose
            ndim: number of spatial dimensions

        Returns:
            List[int]: transposed axes order. Can be passed to np.transpose
                to transpose the axes.
        """
        axes = list(np.array(axes))  # need list to allow shuffle
        assert np.max(axes) <= ndim, "axes must only contain valid axis ids"

        static_axes = list(range(ndim))
        for i in axes:
            static_axes[i] = -1
        np.random.shuffle(axes)

        ctr = 0
        for j, i in enumerate(static_axes):
            if i == -1:
                static_axes[j] = axes[ctr]
                ctr += 1
        return static_axes

    @staticmethod
    def get_matrix(transpose_axes: Sequence[int], ndim: int) -> np.ndarray:
        """
        Create matrix to transpose points

        Args:
            transpose_axes: transposed axes order
            ndim: number of spatial dimensions

        Returns:
            np.ndarray: matrix for transposing in homogeneous coordinates
        """
        mat = np.zeros((ndim + 1, ndim + 1))
        for target_idx, source_idx in transpose_axes:
            mat[source_idx, target_idx] = 1
        return mat


def transpose_array(data: np.ndarray, axes: List[int]) -> np.ndarray:
    """
    Transpose array with channel dim

    Args:
        data: data with channel dim. Expects array in format [C, s_dims] where
            C is the number of color channels and s_dims are spatial dimensions
        axes: transposed axes order

    Returns:
        np.ndarray: transposed array
    """
    _axes = [0] + axes  # add color channel
    return data.transpose(*_axes)


def transpose_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Tranpose points

    Args:
        points: points to transpose. Expects points to be in the format
            [R, L, dims + 1] where  R is the number of objects,
            L is the number of points per object and dims are the number
            of spatial dimensions
        matrix: transpose matrix

    Returns:
        np.ndarray: transposed points (same format as input points)
    """
    if points.size == 0:
        return points
    return points @ matrix.T
