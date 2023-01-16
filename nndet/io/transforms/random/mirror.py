from ast import List

import numpy as np


class Mirror:
    pass


def nd_mirror_matrix(
    cartesian_dims: int,
    mirror_dims: List[int],
    data_shape: List[int],
) -> np.ndarray:
    """
    Create n dimensional matrix to for mirroring

    Args:
        cartesian_dims: number of cartesian dimensions
        mirror_dims: dimensions to mirror
        data_shape: shape of image

    Returns:
        Tensor: matrix for mirroring in homogeneous coordinated,
            [cartesian_dims + 1, cartesian_dims + 1]
    """
    mirror_dims = tuple(mirror_dims)
    data_shape = list(data_shape)

    homogeneous_dims = cartesian_dims + 1
    mat = np.eye(homogeneous_dims, dtype=float)

    # reflection
    mat[[mirror_dims] * 2] = -1

    # add data shape to axis which were reflected
    self_tensor = np.zeros(cartesian_dims, dtype=float)
    index_tensor = np.ndarray(mirror_dims, dtype=int)
    src_tensor = np.ndarray([1] * len(mirror_dims), dtype=float)
    offset_mask = np.put_along_axis(self_tensor, index_tensor, src_tensor, 0)
    mat[:-1, -1] = offset_mask * (np.ndarray(data_shape) - 1)
    return mat


# def mirror_points(
#     points: Sequence[torch.Tensor],
#     dims: Sequence[int],
#     data_shapes: Sequence[Sequence[int]],
# ) -> List[torch.Tensor]:
#     """
#     Mirror points along given dimensions

#     Args:
#         points: points per batch element [N, dims]
#         dims: dimensions to mirror
#         data_shapes: shape of data

#     Returns:
#         Tensor: transformed points [N, dims]
#     """
#     cartesian_dims = points[0].shape[1]
#     homogeneous_points = points_to_homogeneous(points)

#     transformed = []
#     for points_per_image, data_shape in zip(homogeneous_points, data_shapes):
#         matrix = nd_mirror_matrix(cartesian_dims, dims, data_shape).to(points_per_image)
#         transformed.append(points_per_image @ matrix.transpose(0, 1))
#     return points_to_cartesian(transformed)
