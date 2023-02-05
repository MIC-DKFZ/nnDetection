from typing import Tuple
from unittest.mock import Mock, patch

import numpy as np
import pytest
from batchgenerators.augmentations.spatial_transformations import (
    augment_spatial as augmen_spatial_bg,
)

from nndet.io.transforms.detection.spatial import SpatialTransform, augment_spatial


@pytest.fixture
def example() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.zeros((1, 15, 15, 15))
    seg = np.zeros((1, 15, 15, 15))
    seg[0, 9, 9, 9] = 1
    points = np.array([[[9, 9, 9]]])
    return data, seg, points


def test_empty_points(example):
    np.random.seed(0)
    data, seg, points = example
    points = np.array([[[]]]).reshape(0, 0, 3)
    data_result, seg_result, points_result = augment_spatial(
        data=data,
        seg=seg,
        points=points,
        do_elastic_deform=True,
        do_rotation=True,
        do_scale=True,
        patch_size=(12, 12, 12),
    )
    assert points_result.size == 0
    assert points_result.shape == (0, 0, 3)

    data_result, seg_result, points_result = augment_spatial(
        data=data,
        seg=seg,
        points=points,
        do_elastic_deform=False,
        do_rotation=False,
        do_scale=False,
        patch_size=(12, 12, 12),
    )
    assert points_result.size == 0
    assert points_result.shape == (0, 0, 3)


def test_elastic_spatial_fn(example):
    np.random.seed(0)
    data, seg, points = example
    data_result, seg_result, points_result = augment_spatial(
        data=data,
        seg=seg,
        points=points,
        do_elastic_deform=True,
        do_rotation=False,
        do_scale=False,
        patch_size=(15, 15, 15),
        alpha=(0.0, 500.0),
        sigma=(10.0, 13.0),
    )
    _, p0, p1, p2 = np.nonzero(seg_result)
    seg_points = np.stack([p0, p1, p2], axis=-1)[None]
    assert np.allclose(seg_points, np.round(points_result))


def test_elastic_spatial_fn_small_patch(example):
    np.random.seed(0)
    data, seg, points = example
    data_result, seg_result, points_result = augment_spatial(
        data=data,
        seg=seg,
        points=points,
        do_elastic_deform=True,
        do_rotation=False,
        do_scale=False,
        patch_size=(11, 11, 11),
        alpha=(0.0, 500.0),
        sigma=(10.0, 13.0),
    )
    _, p0, p1, p2 = np.nonzero(seg_result)
    seg_points = np.stack([p0, p1, p2], axis=-1)[None]
    assert np.allclose(seg_points, np.round(points_result))


def test_elastic_spatial_fn_large_patch(example):
    np.random.seed(0)
    data, seg, points = example
    data_result, seg_result, points_result = augment_spatial(
        data=data,
        seg=seg,
        points=points,
        do_elastic_deform=True,
        do_rotation=False,
        do_scale=False,
        patch_size=(19, 19, 19),
        alpha=(0.0, 500.0),
        sigma=(10.0, 13.0),
    )
    _, p0, p1, p2 = np.nonzero(seg_result)
    seg_points = np.stack([p0, p1, p2], axis=-1)[None]
    assert np.allclose(seg_points, np.round(points_result))


@patch("nndet.io.transforms.detection.spatial.np.random.uniform", Mock(return_value=1 / 3))
def test_scale_spatial_fn(example):
    data, seg, points = example
    _, seg_result, points_result = augment_spatial(
        data=data,
        seg=seg,
        points=points,
        do_elastic_deform=False,
        do_rotation=False,
        do_scale=True,
        # scale=(0.5, 0.5),
        p_scale_per_sample=1.0,
        patch_size=(11, 11, 11),
    )
    d = 7.0 + (9.0 - 7.0) * 1 / (1 / 3) - 4.0 / 2.0
    expected_points = np.array([[[d, d, d]]], dtype=float)
    assert np.allclose(expected_points, points_result)
    # _, p0, p1, p2 = np.nonzero(seg_result)
    # seg_points = np.stack([p0, p1, p2], axis=-1)[None]
    # assert np.allclose(seg_points, points_result)


def test_rot90_x_spatial_fn(example):
    data, seg, points = example
    _, seg_result, points_result = augment_spatial(
        data=data,
        seg=seg,
        points=points,
        do_elastic_deform=False,
        do_rotation=True,
        do_scale=False,
        patch_size=(11, 11, 11),
        angle_x=(np.pi / 2, np.pi / 2),
        angle_y=(0, 0),
        angle_z=(0, 0),
        p_rot_per_sample=1.0,
    )
    expected_points = np.array([[[7.0, 3.0, 7.0]]], dtype=float)
    assert np.allclose(expected_points, points_result)
    _, p0, p1, p2 = np.nonzero(seg_result)
    seg_points = np.stack([p0, p1, p2], axis=-1)[None]
    assert np.allclose(seg_points, points_result)


def test_rot90_y_spatial_fn(example):
    data, seg, points = example
    _, seg_result, points_result = augment_spatial(
        data=data,
        seg=seg,
        points=points,
        do_elastic_deform=False,
        do_rotation=True,
        do_scale=False,
        patch_size=(11, 11, 11),
        angle_x=(0, 0),
        angle_y=(np.pi / 2, np.pi / 2),
        angle_z=(0, 0),
        p_rot_per_sample=1.0,
    )
    expected_points = np.array([[[7.0, 7.0, 3.0]]], dtype=float)
    assert np.allclose(expected_points, points_result)
    _, p0, p1, p2 = np.nonzero(seg_result)
    seg_points = np.stack([p0, p1, p2], axis=-1)[None]
    assert np.allclose(seg_points, points_result)


def test_rot90_z_spatial_fn(example):
    data, seg, points = example
    _, seg_result, points_result = augment_spatial(
        data=data,
        seg=seg,
        points=points,
        do_elastic_deform=False,
        do_rotation=True,
        do_scale=False,
        patch_size=(11, 11, 11),
        angle_x=(0, 0),
        angle_y=(0, 0),
        angle_z=(np.pi / 2, np.pi / 2),
        p_rot_per_sample=1.0,
    )
    # 5, 9, 9 for p=15
    expected_points = np.array([[[3.0, 7.0, 7.0]]], dtype=float)
    assert np.allclose(expected_points, points_result)
    _, p0, p1, p2 = np.nonzero(seg_result)
    seg_points = np.stack([p0, p1, p2], axis=-1)[None]
    assert np.allclose(seg_points, points_result)


def test_rot90_z_bigger_p_spatial_fn(example):
    data, seg, points = example
    _, seg_result, points_result = augment_spatial(
        data=data,
        seg=seg,
        points=points,
        do_elastic_deform=False,
        do_rotation=True,
        do_scale=False,
        patch_size=(19, 19, 19),
        angle_x=(0, 0),
        angle_y=(0, 0),
        angle_z=(np.pi / 2, np.pi / 2),
        p_rot_per_sample=1.0,
    )
    # 5, 9, 9 for p=15
    expected_points = np.array([[[7.0, 11.0, 11.0]]], dtype=float)
    assert np.allclose(expected_points, points_result)
    _, p0, p1, p2 = np.nonzero(seg_result)
    seg_points = np.stack([p0, p1, p2], axis=-1)[None]
    assert np.allclose(seg_points, points_result)


def test_crop_outside_spatial_fn(example):
    data, seg, points = example
    _, seg_result, points_result = augment_spatial(
        data=data,
        seg=seg,
        points=points,
        do_elastic_deform=False,
        do_rotation=False,
        do_scale=False,
        patch_size=(3, 3, 3),
    )
    expected_points = np.array([[[3.0, 3.0, 3.0]]], dtype=float)
    assert np.allclose(expected_points, points_result)
    _, p0, p1, p2 = np.nonzero(seg_result)
    seg_points = np.stack([p0, p1, p2], axis=-1)[None]
    assert np.allclose(seg_points, points_result)


def test_crop_spatial_fn(example):
    data, seg, points = example
    _, seg_result, points_result = augment_spatial(
        data=data,
        seg=seg,
        points=points,
        do_elastic_deform=False,
        do_rotation=False,
        do_scale=False,
        patch_size=(5, 5, 5),
    )
    expected_points = np.array([[[4.0, 4.0, 4.0]]], dtype=float)
    assert np.allclose(expected_points, points_result)
    _, p0, p1, p2 = np.nonzero(seg_result)
    seg_points = np.stack([p0, p1, p2], axis=-1)[None]
    assert np.allclose(seg_points, points_result)


def test_crop_pad_spatial_fn(example):
    data, seg, points = example
    _, seg_result, points_result = augment_spatial(
        data=data,
        seg=seg,
        points=points,
        do_elastic_deform=False,
        do_rotation=False,
        do_scale=False,
        patch_size=(23, 23, 23),
    )
    expected_points = np.array([[[13.0, 13.0, 13.0]]], dtype=float)
    assert np.allclose(expected_points, points_result)
    _, p0, p1, p2 = np.nonzero(seg_result)
    seg_points = np.stack([p0, p1, p2], axis=-1)[None]
    assert np.allclose(seg_points, points_result)


def test_crop_same_spatial_fn(example):
    data, seg, points = example
    _, seg_result, points_result = augment_spatial(
        data=data,
        seg=seg,
        points=points,
        do_elastic_deform=False,
        do_rotation=False,
        do_scale=False,
        patch_size=(15, 15, 15),
    )
    expected_points = np.array([[[9.0, 9.0, 9.0]]], dtype=float)
    assert np.allclose(expected_points, points_result)
    _, p0, p1, p2 = np.nonzero(seg_result)
    seg_points = np.stack([p0, p1, p2], axis=-1)[None]
    assert np.allclose(seg_points, points_result)


def test_spatial_nndet_bg():
    # test against bg spatial augment
    n = 300
    for i in range(n):
        np.random.seed(i + n)
        data = np.random.rand(1, 1, 15, 15, 15)
        seg = np.zeros((1, 1, 15, 15, 15))
        seg[0, 0, 7, 7, 7] = 1
        seg[0, 0, 12, 14, 7] = 2
        seg[0, 0, 2:4, 4:8, 1:5] = 3

        # test different patch sizes
        if i % 3 == 0:
            p = 18
        elif i % 2 == 0:
            p = 12
        else:
            p = 15

        # bg original code
        np.random.seed(i)
        data_result_bg, seg_result_bg = augmen_spatial_bg(
            data=data,
            seg=seg,
            do_elastic_deform=True,
            do_rotation=True,
            do_scale=True,
            patch_size=(p, p, p),
            random_crop=False,
        )

        # nndet version
        np.random.seed(i)
        data_out = np.zeros((1, 1, p, p, p), dtype=float)
        seg_out = np.zeros((1, 1, p, p, p), dtype=float)
        _, _, _ = augment_spatial(
            data=data[0],
            seg=seg[0],
            do_elastic_deform=True,
            do_rotation=True,
            do_scale=True,
            patch_size=(p, p, p),
            data_out=data_out[0],
            seg_out=seg_out[0],
        )
        assert data_result_bg.shape == data_out.shape
        assert seg_result_bg.shape == seg_out.shape
        assert np.allclose(data_result_bg, data_out)
        assert np.allclose(seg_result_bg, seg_out)


def test_spatial_transform_smoke():
    points = np.array(
        [
            [-1, -1, -1, 1],
            [-1, -1, 1, 1],
            [-1, 1, -1, 1],
            [-1, 1, 1, 1],
            [1, -1, -1, 1],
            [1, -1, 1, 1],
            [1, 1, -1, 1],
            [1, 1, 1, 1],
        ],
        dtype=float,
    )  # [1, 8, 3 + 1]
    batch = {
        "data": np.random.rand(2, 1, 16, 16, 16),
        "seg": np.random.rand(2, 1, 16, 16, 16),
        "point": [points, points],
    }
    trafo = SpatialTransform(
        patch_size=(8, 8, 8),
        data_key="data",
        label_key="seg",
        point_key="point",
        do_elastic_deform=True,
        do_rotation=True,
        do_scale=True,
    )
    result = trafo(**batch)
