from typing import Hashable, Optional, Sequence, Tuple, Union

import numpy as np
from batchgenerators.augmentations.utils import (
    create_matrix_rotation_2d,
    create_matrix_rotation_x_3d,
    create_matrix_rotation_y_3d,
    create_matrix_rotation_z_3d,
    interpolate_img,
)
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import nndet.core.ops_np as ops_np
from nndet.io.transforms.random.crop_bg import crop


class SpatialTransform(AbstractTransform):
    def __init__(
        self,
        patch_size: Sequence[int],
        data_key: Hashable,
        label_key: Optional[Hashable] = None,
        point_key: Optional[Hashable] = None,
        # elastic
        do_elastic_deform: bool = True,
        p_el_per_sample: float = 1,
        alpha: Tuple[float, float] = (0.0, 1000.0),
        sigma: Tuple[float, float] = (10.0, 13.0),
        # rotation
        do_rotation: bool = True,
        p_rot_per_sample: float = 1,
        angle_x: Tuple[float, float] = (0, 2 * np.pi),
        angle_y: Tuple[float, float] = (0, 2 * np.pi),
        angle_z: Tuple[float, float] = (0, 2 * np.pi),
        p_rot_per_axis: float = 1,
        # scaling
        do_scale: bool = True,
        p_scale_per_sample: float = 1,
        scale: Tuple[float, float] = (0.75, 1.25),
        independent_scale_for_each_axis: bool = False,
        p_independent_scale_per_axis: int = 1,
        # interpolation and cropping
        border_mode_data: str = "nearest",
        border_cval_data: int = 0,
        order_data: int = 3,
        border_mode_seg: str = "constant",
        border_cval_seg: int = 0,
        order_seg: int = 0,
        # misc
        random_crop: bool = False,
        patch_center_dist_from_border: int = 30,
    ):
        """
        The ulatimate augmentation function :)
        Performs elastic, rotation, scaling and cropping in one transformation
        to reduce the number of interpolation steps. It supports data, segmentations
        and point annotations.

        Args:
            patch_size: final patch size for cropping
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
            do_elastic_deform: If enabled, perform elastic deformation. Elastic
                deformation is performed by generating a random offset
                field which is smoothed by an guassian kernel (kernel size gamma)
                and scaled by alpha. Defaults to True.
            p_el_per_sample: probability to apply elastic deformation to a single
                sample. Defaults to 1.
            alpha: Magnitude range of deformation field. Defaults to (0.0, 1000.0).
            sigma: Standard deviation range of gaussian filter. See scikit image
                gaussian filter for more info. Defaults to (10.0, 13.0).
            do_rotation: If enable perform random rotation aroud axes. Rotation
                order: x->y->z. Rotations are not uniformly sampled
                https://github.com/MIC-DKFZ/batchgenerators/issues/84.
                Defaults to True.
            p_rot_per_sample: probability to apply rotation to a single sample.
                Defaults to 1.
            p_rot_per_axis: probability that a single axis will be rotated.
                Defaults to 1.
            angle_x: Rotation range in radian. Defaults to (0, 2 * np.pi).
            angle_y: Rotation range in radian.. Defaults to (0, 2 * np.pi).
            angle_z: Rotation range in radian.. Defaults to (0, 2 * np.pi).
            do_scale: if enabled random scaling is applied. Defaults to True.
            p_scale_per_sample: probability to scale a single sample. Defaults to 1.
            scale: Scale range. Values lower and above one are sampled separately.
                Defaults to (0.75, 1.25).
            p_independent_scale_per_axis: Enable scaling of axis individually.
                Only used if `independent_scale_for_each_axis` is `True`.
                Defaults to 1.
            independent_scale_for_each_axis: Probability to scale each axis.
                Defaults to False.
            order_data: order to interpolate data passed to map_coordiantes.
                See scikit image map_coordinates coordinated for more info.
                Defaults to 3.
            border_cval_data: border value passed to map_coordinates.
                See scikit image map_coordinates coordinated for more info.
                Defaults to 0.
            border_mode_data: border mode passed to map_coordinates.
                See scikit image map_coordinates coordinated for more info.
                Defaults to "nearest".
            order_seg: order to interpolate seg passed to map_coordiantes.
                See scikit image map_coordinates coordinated for more info.
                Defaults to 0.
            border_cval_seg: border value passed to map_coordinates.
                See scikit image map_coordinates coordinated for more info.
                Defaults to 0.
            border_mode_seg: border mode passed to map_coordinates.
                See scikit image map_coordinates coordinated for more info.
                Defaults to "constant".
            seg: provide segmentation to augment. [C, dims] where C is the number
                of channels and dims are spatial dimensions. Defaults to None.
            points: points to augment. Note: in case of elastic_deformation
                the correct poisition of non interger points can only be
                interpolated. [N, R, #dims] where N is the number of objects,
                R is the number of points per object and #dims are the number
                of spatial dimensions
            clip_points: optionally clip points to patch size
            random_crop: Not supported here. Defaults to False.
            patch_center_dist_from_border: Not used here. Defaults to 30.

        Raises:
            NotImplementedError: if random_crop is enabled.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]: returns
                a tuple with three entries: data, seg and points. All of them
                follow the same format as their input equivalent and are None
                if not provided as input.

        Warning:
            Elastic deformation on points is an experimental feature and
            requires additional testing! Use on your own risk.
        """
        # keys
        self.data_key = data_key
        self.label_key = label_key
        self.point_key = point_key
        self.patch_size = patch_size

        # elastic
        self.do_elastic_deform = do_elastic_deform
        self.p_el_per_sample = p_el_per_sample
        self.alpha = alpha
        self.sigma = sigma

        # rotation
        self.do_rotation = do_rotation
        self.p_rot_per_sample = p_rot_per_sample
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.p_rot_per_axis = p_rot_per_axis

        # scaling
        self.do_scale = do_scale
        self.p_scale_per_sample = p_scale_per_sample
        self.scale = scale
        self.independent_scale_for_each_axis = independent_scale_for_each_axis
        self.p_independent_scale_per_axis = p_independent_scale_per_axis

        # interpolation
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.order_seg = order_seg

        # misc
        self.random_crop = random_crop
        self.patch_center_dist_from_border = patch_center_dist_from_border

    def __call__(self, **data_dict):
        # inputs
        data = data_dict[self.data_key]
        batch_size = len(data)
        if self.label_key is not None:
            seg = data_dict[self.label_key]
        else:
            seg = None
        if self.point_key is not None:
            points = data_dict[self.point_key]
        else:
            points = None

        # checks
        assert len(self.patch_size) == data.ndim - 2

        data_result = np.zeros((data.shape[0], data.shape[1], *self.patch_size), dtype=float)
        if seg is not None:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], *self.patch_size), dtype=float)
        points_result = []

        # apply
        for b in range(batch_size):
            if points is not None:
                p = ops_np.points_to_cartesian(points)

            # data and seg directly saved into arrays
            _, _, point_sample = augment_spatial(
                # inputs
                data=data[b],
                seg=seg[b] if seg is not None else None,
                points=p[b] if points is not None else None,
                data_out=data_result[b],
                seg_out=seg_result[b],
                patch_size=self.patch_size,
                # elastic
                do_elastic_deform=self.do_elastic_deform,
                p_el_per_sample=self.p_el_per_sample,
                alpha=self.alpha,
                sigma=self.sigma,
                # rotation
                do_rotation=self.do_rotation,
                p_rot_per_sample=self.p_rot_per_sample,
                angle_x=self.angle_x,
                angle_y=self.angle_y,
                angle_z=self.angle_z,
                p_rot_per_axis=self.p_rot_per_axis,
                # scale
                do_scale=self.do_scale,
                p_scale_per_sample=self.p_scale_per_sample,
                scale=self.scale,
                independent_scale_for_each_axis=self.independent_scale_for_each_axis,
                p_independent_scale_per_axis=self.p_independent_scale_per_axis,
                # interpolation
                border_mode_data=self.border_mode_data,
                border_cval_data=self.border_cval_data,
                order_data=self.order_data,
                border_mode_seg=self.border_mode_seg,
                border_cval_seg=self.border_cval_seg,
                order_seg=self.order_seg,
                # misc
                random_crop=self.random_crop,
                patch_center_dist_from_border=self.patch_center_dist_from_border,
            )
            if points is not None:
                points_result.append(ops_np.points_to_homogeneous([point_sample])[0])

        # save batch
        data_dict[self.data_key] = data_result
        if seg is not None:
            data_dict[self.label_key] = seg_result
        if points is not None:
            data_dict[self.point_key] = points_result
        return data_dict


class CoordinateMapper:
    def __init__(self, img_shape: Sequence[int], patch_size: Sequence[int]) -> None:
        """
        Helper class to create coordinate meshes and norm/denorm points

        Args:
            patch_size: size of final patch after cropping
        """
        self.img_shape = img_shape
        self.img_shape_array = np.array(img_shape)
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
        offset = (self.img_shape_array - 1.0) / 2.0
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
        return points - ((self.img_shape_array - 1.0) / 2.0)[None, None]

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
    # scale
    do_scale: bool = True,
    p_scale_per_sample: float = 1,
    scale: Tuple[float, float] = (0.75, 1.25),
    independent_scale_for_each_axis: bool = False,
    p_independent_scale_per_axis: int = 1,
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
    data_out: Optional[np.ndarray] = None,  # [C, dims]
    seg_out: Optional[np.ndarray] = None,  # [C, dims]
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
        p_el_per_sample: probability to apply elastic deformation to a single
            sample. Defaults to 1.
        alpha: Magnitude range of deformation field. Defaults to (0.0, 1000.0).
        sigma: Standard deviation range of gaussian filter. See scikit image
            gaussian filter for more info. Defaults to (10.0, 13.0).
        do_rotation: If enable perform random rotation aroud axes. Rotation
            order: x->y->z. Rotations are not uniformly sampled
            https://github.com/MIC-DKFZ/batchgenerators/issues/84.
            Defaults to True.
        p_rot_per_sample: probability to apply rotation to a single sample.
            Defaults to 1.
        p_rot_per_axis: probability that a single axis will be rotated.
            Defaults to 1.
        angle_x: Rotation range in radian. Defaults to (0, 2 * np.pi).
        angle_y: Rotation range in radian.. Defaults to (0, 2 * np.pi).
        angle_z: Rotation range in radian.. Defaults to (0, 2 * np.pi).
        do_scale: if enabled random scaling is applied. Defaults to True.
        p_scale_per_sample: probability to scale a single sample. Defaults to 1.
        scale: Scale range. Values lower and above one are sampled separately.
            Defaults to (0.75, 1.25).
        p_independent_scale_per_axis: Enable scaling of axis individually.
            Only used if `independent_scale_for_each_axis` is `True`.
            Defaults to 1.
        independent_scale_for_each_axis: Probability to scale each axis.
            Defaults to False.
        order_data: order to interpolate data passed to map_coordiantes.
            See scikit image map_coordinates coordinated for more info.
             Defaults to 3.
        border_cval_data: border value passed to map_coordinates.
            See scikit image map_coordinates coordinated for more info.
            Defaults to 0.
        border_mode_data: border mode passed to map_coordinates.
            See scikit image map_coordinates coordinated for more info.
            Defaults to "nearest".
        order_seg: order to interpolate seg passed to map_coordiantes.
            See scikit image map_coordinates coordinated for more info.
            Defaults to 0.
        border_cval_seg: border value passed to map_coordinates.
            See scikit image map_coordinates coordinated for more info.
            Defaults to 0.
        border_mode_seg: border mode passed to map_coordinates.
            See scikit image map_coordinates coordinated for more info.
            Defaults to "constant".
        seg: provide segmentation to augment. [C, dims] where C is the number
            of channels and dims are spatial dimensions. Defaults to None.
        points: points to augment. Note: in case of elastic_deformation
            the correct poisition of non interger points can only be
            interpolated. [N, R, #dims] where N is the number of objects,
            R is the number of points per object and #dims are the number
            of spatial dimensions
        data_out: provide pre initialized array to save results into [C, dims]
            where C is the number of channels and dims are spatial dimensions.
            Defaults to None.
        seg_out: provide pre initialized array to save results into [C, dims]
            where C is the number of channels and dims are spatial dimensions.
            Defaults to None.
        clip_points: optionally clip points to patch size
        random_crop: Not supported here. Defaults to False.
        patch_center_dist_from_border: Not used here. Defaults to 30.

    Raises:
        NotImplementedError: if random_crop is enabled.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]: returns
            a tuple with three entries: data, seg and points. All of them
            follow the same format as their input equivalent and are None
            if not provided as input.

    Warning:
        Elastic deformation on points is an experimental feature and
        requires additional testing! Use on your own risk.
    """
    if random_crop:
        raise NotImplementedError("Random crop is not implemented")
    dim = len(patch_size)
    mapper = CoordinateMapper(img_shape=data.shape[1:], patch_size=patch_size)

    if data_out is None:
        data_out = np.zeros((*data.shape[:1], *patch_size), dtype=float)
    else:
        assert tuple(data_out.shape) == (*data.shape[:1], *patch_size)

    if seg is not None:
        if seg_out is None:
            seg_out = np.zeros((*seg.shape[:1], *patch_size), dtype=float)
        else:
            assert tuple(seg_out.shape) == (*seg.shape[:1], *patch_size)

    point_result = None

    # create coordinate grid
    coords = mapper.zero_center_coords()
    points_empty = points.size == 0 if points is not None else False
    # Note: always check points_sample against None to cover empty and no points
    points_sample = None if points is None or points_empty else mapper.zero_center_points(points)
    modified_coords = False

    # perform augmentations
    # elastic
    if do_elastic_deform and np.random.uniform() < p_el_per_sample:
        coords, points_sample = apply_elastic_deform(
            coords,
            alpha=alpha,
            sigma=sigma,
            points=points_sample,
            points_img=mapper.img_origin_center_points(points_sample) if points_sample is not None else None,
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
        if points_sample is not None:
            points_sample = mapper.img_origin_center_points(points_sample)

        for channel_id in range(data.shape[0]):
            data_out[channel_id] = interpolate_img(
                data[channel_id],
                coords,
                order_data,
                border_mode_data,
                cval=border_cval_data,
            )
        if seg is not None:
            for channel_id in range(seg.shape[0]):
                seg_out[channel_id] = interpolate_img(
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
            points=[points] if points_sample is not None else None,
        )
        data_out = d[0]
        if seg is not None:
            seg_out = s[0]
        if points_sample is not None:
            points_sample = p[0]

    if clip_points and points_sample is not None:
        points_sample = np.clip(points_sample, a_min=0, a_max=np.array(patch_size)[None, None])

    if points_empty:
        point_result = points  # empty array
    else:
        point_result = points_sample
    return data_out, seg_out, point_result


def apply_elastic_deform(
    coords: np.ndarray,
    alpha: Tuple[float, float],
    sigma: Tuple[float, float],
    points: Optional[np.ndarray] = None,
    points_img: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply random elastic deformation to coordinate mesh and points

    Args:
        coords: coordinate mesh
        alpha: alpha specifying the magnitdue range
        sigma: sigma specifying the standard deviation range of the
            gaussian kernel
        points: points in coordiante mesh frame. Defaults to None.
        points_img: points in original image frame. No augmentations should
            have been applied befor the elastic augmentation! Defaults to None.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: coordiante mesh and points

    Warning:
        No other transformations are allowed before running this on point
        coordinates!
    """
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
    """
    Perform elastic deformation to coordinate mesh and points

    Args:
        coords: coordinate mesh
        alpha: alpha specifying the magnitdue
        sigma: sigma specifying the standard deviation
        points: points in coordiante mesh frame. Defaults to None.
        points_img: points in original image frame. No augmentations should
            have been applied befor the elastic augmentation! Defaults to None.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: coordiante mesh and points
    """
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
    """
    Apply random rotation to coordinate mesh and points

    Args:
        coords: coordinate mesh
        dim: number of spatial dimensions
        angle_x: rotation range in radian
        angle_y: rotation range in radian
        angle_z: rotation range in radian
        p_rot_per_axis: probability to apply rotation to an axis
        points: points in coordiante mesh frame. Defaults to None.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: coordiante mesh and points
    """
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
    """
    Perform rotation

    Args:
        coords: coordinate mesh
        angles: rotation angles. In 2D this is a float, in 3D this should be
            a tuple of three float each specifying the rotation angle for
            the axis. The first entry corresponds to the x axis, the last one
            to the z axis.
        points: points in coordiante mesh frame. Defaults to None.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: coordiante mesh and points
    """
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
    """
    Apply random scaling to coordinate mesh and points

    Args:
        coords: coordinate mesh
        dim: number of spatial dimensions
        scale: scaling range.
        independent_scale_for_each_axis: if enabled, different scaling values
            can be applied to different axis.
        p_independent_scale_per_axis: Probability to apply rotation to an
            axis. Only used if `independent_scale_for_each_axis` is `True`.
        points: points in coordiante mesh frame. Defaults to None.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: coordiante mesh and points
    """
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
    """
    Perform scaling of mesh and points

    Args:
        coords: coordinate mesh
        scale: scale value
        points: points in coordiante mesh frame. Defaults to None.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: coordiante mesh and points
    """
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
