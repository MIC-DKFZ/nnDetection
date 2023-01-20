from typing import Optional, Sequence, Tuple, Union

import numpy as np
from batchgenerators.augmentations.utils import (
    create_matrix_rotation_2d,
    create_matrix_rotation_x_3d,
    create_matrix_rotation_y_3d,
    create_matrix_rotation_z_3d,
    interpolate_img,
)
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from nndet.io.transforms.random.crop_bg import crop


class CoordinateMapper:
    def __init__(self, patch_size: Sequence[int]) -> None:
        """
        Helper class to create coordinate meshes and norm/denorm points

        Args:
            patch_size: size of final patch after cropping
        """
        self.patch_size = patch_size
        self.patch_size_array = np.array(patch_size, dtype=float)
        self.dim = len(patch_size)

    def zero_center_coords(self) -> np.ndarray:
        """
        Create zero centered coordinate mesh compatible

        Returns:
            np.ndarray: zero center coordinate mesh [#dims, dims] where #dims
                is the number of spatial dimensions and dims are the spatial
                dimensions
        """
        tmp = tuple([np.arange(i) for i in self.patch_size])
        coords = np.array(np.meshgrid(*tmp, indexing="ij")).astype(float)
        offset = (self.patch_size_array - 1) / 2.0
        return coords - np.expand_dims(offset, axis=tuple(range(1, self.dim + 1)))

    def img_origin_center_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Move zero centered coordinate mesh into image origin. Everything that
        remains <0 is outside of the image now.

        Returns:
            np.ndarray: image centered coordinate mesh [#dims, dims] where #dims
                is the number of spatial dimensions and dims are the spatial
                dimensions
        """
        offset = (self.patch_size_array - 1.0) / 2.0
        return coords + np.expand_dims(offset, axis=tuple(range(1, self.dim + 1)))

    def zero_center_points(self, points: np.ndarray) -> np.ndarray:
        """
        Zero center points for augmentation

        Args:
            points: image points [N, R, #dims] where N is the number of objects,
                R is the number of points per objects and #dims is the number
                of spatial dimensions

        Returns:
            np.ndarray: zero centered points, same format as input points
        """
        return points - ((self.patch_size_array - 1.0) / 2.0)[None, None]

    def img_origin_center_points(self, points: np.ndarray) -> np.ndarray:
        """
        Move points into original image frame

        Args:
            points: zero centered points [N, R, #dims] where N is the number
                of objects, R is the number of points per objects and #dims
                is the number of spatial dimensions

        Returns:
            np.ndarray: points in image frame, same format as input points
        """
        return points + ((self.patch_size_array - 1.0) / 2.0)[None, None]


def augment_spatial(
    data: np.ndarray,
    patch_size: Sequence[int],
    # elastic
    do_elastic_deform: bool = True,
    p_el_per_sample: float = 1,
    alpha: Tuple[float, float] = (0.0, 1000.0),
    sigma: Tuple[float, float] = (10.0, 13.0),
    # rotation
    do_rotation: bool = True,
    p_rot_per_sample: float = 1,
    p_rot_per_axis: float = 1,
    angle_x: Tuple[float, float] = (0, 2 * np.pi),
    angle_y: Tuple[float, float] = (0, 2 * np.pi),
    angle_z: Tuple[float, float] = (0, 2 * np.pi),
    independent_scale_for_each_axis: bool = False,
    p_independent_scale_per_axis: int = 1,
    # scale
    do_scale: bool = True,
    p_scale_per_sample: float = 1,
    scale: Tuple[float, float] = (0.75, 1.25),
    # interpolation & padding
    order_data: int = 3,
    border_cval_data: int = 0,
    border_mode_data: str = "nearest",
    order_seg: int = 0,
    border_cval_seg: int = 0,
    border_mode_seg: str = "constant",
    # inputs
    seg: Optional[np.ndarray] = None,  # [C, dims]
    points: Optional[np.ndarray] = None,  # [N, R, dims]
    clip_points: bool = False,
    # not supported
    random_crop: bool = False,
    patch_center_dist_from_border: int = 30,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    The ulatimate augmentation function :)
    Performs elastic, rotation, scaling and cropping in one transformation
    to reduce the number of interpolation steps. It supports data, segmentations
    and point annotations.

    Args:
        data: input data
        patch_size: final patch size for cropping
        do_elastic_deform: If enabled, perform elastic deformation. Elastic
            deformation is performed by generating a random offset
            field which is smoothed by an guassian kernel (kernel size gamma)
            and scaled by alpha. Defaults to True.
        p_el_per_sample: _description_. Defaults to 1.
        alpha: _description_. Defaults to (0.0, 1000.0).
        sigma: _description_. Defaults to (10.0, 13.0).
        do_rotation: _description_. Defaults to True.
        p_rot_per_sample: _description_. Defaults to 1.
        p_rot_per_axis: _description_. Defaults to 1.
        angle_x: _description_. Defaults to (0, 2 * np.pi).
        angle_y: _description_. Defaults to (0, 2 * np.pi).
        angle_z: _description_. Defaults to (0, 2 * np.pi).
        independent_scale_for_each_axis: _description_. Defaults to False.
        p_independent_scale_per_axis: _description_. Defaults to 1.
        do_scale: _description_. Defaults to True.
        p_scale_per_sample: _description_. Defaults to 1.
        scale: _description_. Defaults to (0.75, 1.25).
        order_data: _description_. Defaults to 3.
        border_cval_data: _description_. Defaults to 0.
        border_mode_data: _description_. Defaults to "nearest".
        order_seg: _description_. Defaults to 0.
        border_cval_seg: _description_. Defaults to 0.
        border_mode_seg: _description_. Defaults to "constant".
        seg: _description_. Defaults to None.
        patch_center_dist_from_border: _description_. Defaults to 30.

    Raises:
        NotImplementedError: _description_

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]: _description_
    """
    # TODO: speed up if points is empty
    if random_crop:
        raise NotImplementedError("Random crop is not implemented")
    dim = len(patch_size)
    mapper = CoordinateMapper(patch_size)
    data_result = np.zeros((*data.shape[:1], *patch_size), dtype=np.float32)
    if seg is not None:
        seg_result = np.zeros((*data.shape[:1], *patch_size), dtype=np.float32)
    else:
        seg_result = None
    point_result = None

    # create coordinate grid
    coords = mapper.zero_center_coords()
    points_sample = mapper.zero_center_points(points) if points is not None else None
    modified_coords = False

    # perform augmentations
    # elastic
    if do_elastic_deform and np.random.uniform() < p_el_per_sample:
        coords, points_sample = apply_elastic_deform(
            coords,
            alpha=alpha,
            sigma=sigma,
            points=points_sample,
            points_img=points,
        )
        modified_coords = True

    # rotation
    if do_rotation and np.random.uniform() < p_rot_per_sample:
        coords, points_sample = apply_rotation(
            coords,
            dim=dim,
            angle_x=angle_x,
            angle_y=angle_y,
            angle_z=angle_z,
            p_rot_per_axis=p_rot_per_axis,
            points=points_sample,
        )
        modified_coords = True

    # scale
    if do_scale and np.random.uniform() < p_scale_per_sample:
        coords, points_sample = apply_scale(
            coords,
            dim=dim,
            scale=scale,
            independent_scale_for_each_axis=independent_scale_for_each_axis,
            p_independent_scale_per_axis=p_independent_scale_per_axis,
            points=points_sample,
        )
        modified_coords = True

    if modified_coords:
        coords = mapper.img_origin_center_coords(coords)
        if points is not None:
            points_sample = mapper.img_origin_center_points(points_sample)

        for channel_id in range(data.shape[0]):
            data_result[channel_id] = interpolate_img(
                data[channel_id],
                coords,
                order_data,
                border_mode_data,
                cval=border_cval_data,
            )
        if seg is not None:
            for channel_id in range(seg.shape[0]):
                seg_result[channel_id] = interpolate_img(
                    seg[channel_id],
                    coords,
                    order_seg,
                    border_mode_seg,
                    cval=border_cval_seg,
                    is_seg=True,
                )
    else:
        # perform cropping
        d, s, p = crop(
            data=data[None],
            seg=None if seg is None else seg[None],
            crop_size=patch_size,
            margins=0,
            crop_type="center",
            points=[points_sample] if points_sample is not None else None,
        )
        data_result = d[0]
        if seg is not None:
            seg_result = s[0]
        if points is not None:
            points_sample = mapper.img_origin_center_points(p[0])

    if clip_points and points is not None:
        points_sample = np.clip(points_sample, a_min=0, a_max=np.array(patch_size)[None, None])
    point_result = points_sample
    return data_result, seg_result, point_result


def apply_elastic_deform(
    coords: np.ndarray,
    alpha: Tuple[float, float],
    sigma: Tuple[float, float],
    points: Optional[np.ndarray] = None,
    points_img: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    a = np.random.uniform(alpha[0], alpha[1])
    s = np.random.uniform(sigma[0], sigma[1])
    return elastic_deform_coords_points(
        coords,
        alpha=a,
        sigma=s,
        points=points,
        points_img=points_img,
    )


def elastic_deform_coords_points(
    coords: np.ndarray,
    alpha: float,
    sigma: float,
    points: Optional[np.ndarray] = None,
    points_img: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    n_dim = len(coords)
    offsets = []
    for _ in range(n_dim):
        offsets.append(
            gaussian_filter(
                (np.random.random(coords.shape[1:]) * 2 - 1),
                sigma,
                mode="constant",
                cval=0,
            )
            * alpha
        )
    offsets = np.array(offsets)

    coords_new = offsets + coords
    if points is not None:
        points_img_transposed = points_img.transpose(2, 0, 1)  # #dims, N, R
        offsets_points = [map_coordinates(offsets[idx], points_img_transposed, order=1) for idx in range(len(offsets))]
        offsets_points = np.stack(offsets_points, axis=-1)  # N, R, #dims
        points_new = points - offsets_points
    else:
        points_new = None
    return coords_new, points_new


def apply_rotation(
    coords: np.ndarray,
    dim: int,
    angle_x: Tuple[float, float],
    angle_y: Tuple[float, float],
    angle_z: Tuple[float, float],
    p_rot_per_axis: float,
    points: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if np.random.uniform() <= p_rot_per_axis:
        a_x = np.random.uniform(angle_x[0], angle_x[1])
    else:
        a_x = 0

    if dim == 3:
        if np.random.uniform() <= p_rot_per_axis:
            a_y = np.random.uniform(angle_y[0], angle_y[1])
        else:
            a_y = 0

        if np.random.uniform() <= p_rot_per_axis:
            a_z = np.random.uniform(angle_z[0], angle_z[1])
        else:
            a_z = 0

        coords, points = rotate_coords_points(coords, (a_x, a_y, a_z), points)
    else:
        coords, points = rotate_coords_points(coords, a_x, points)
    return coords, points


def rotate_coords_points(
    coords: np.ndarray,
    angles: Union[float, Tuple[float, float, float]],
    points: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if len(coords) == 3:
        rot_matrix = np.identity(len(coords))
        rot_matrix = create_matrix_rotation_x_3d(angles[0], rot_matrix)
        rot_matrix = create_matrix_rotation_y_3d(angles[1], rot_matrix)
        rot_matrix = create_matrix_rotation_z_3d(angles[2], rot_matrix)

        rot_matrix_inv = np.identity(len(coords))
        rot_matrix_inv = create_matrix_rotation_z_3d(-angles[2], rot_matrix_inv)
        rot_matrix_inv = create_matrix_rotation_y_3d(-angles[1], rot_matrix_inv)
        rot_matrix_inv = create_matrix_rotation_x_3d(-angles[0], rot_matrix_inv)
    else:
        rot_matrix = create_matrix_rotation_2d(angles)
        rot_matrix_inv = create_matrix_rotation_2d(-angles)

    coords_new = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
    if points is not None:
        points_new = points @ rot_matrix_inv
    else:
        points_new = None
    return coords_new, points_new


def apply_scale(
    coords: np.ndarray,
    dim: int,
    scale: Tuple[float, float],
    independent_scale_for_each_axis: bool,
    p_independent_scale_per_axis: float,
    points: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if independent_scale_for_each_axis and np.random.uniform() < p_independent_scale_per_axis:
        sc = []
        for _ in range(dim):
            if np.random.random() < 0.5 and scale[0] < 1:
                sc.append(np.random.uniform(scale[0], 1))
            else:
                sc.append(np.random.uniform(max(scale[0], 1), scale[1]))
    else:
        if np.random.random() < 0.5 and scale[0] < 1:
            sc = np.random.uniform(scale[0], 1)
        else:
            sc = np.random.uniform(max(scale[0], 1), scale[1])
    return scale_coords_points(coords, sc, points)


def scale_coords_points(
    coords: np.ndarray,
    scale: Union[float, Sequence[float]],
    points: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if isinstance(scale, (tuple, list, np.ndarray)):
        assert len(scale) == len(coords)
        scale_array = np.array(scale)
    else:
        scale_array = np.array([scale] * len(coords))

    coords_new = coords * np.expand_dims(scale_array, tuple(range(1, len(coords) + 1)))
    if points is not None:
        points_new = points * (1 / scale_array[None, None])
    else:
        points_new = None
    return coords_new, points_new
