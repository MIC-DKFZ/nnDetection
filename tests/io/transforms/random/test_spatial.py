from typing import Tuple
from unittest.mock import Mock, patch

import numpy as np
import pytest
from batchgenerators.augmentations.spatial_transformations import (
    augment_spatial as augmen_spatial_bg,
)

from nndet.io.transforms.random.spatial import augment_spatial


@pytest.fixture
def example() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.zeros((1, 15, 15, 15))
    seg = np.zeros((1, 15, 15, 15))
    seg[0, 9, 9, 9] = 1
    points = np.array([[[9, 9, 9]]])
    return data, seg, points


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
        alpha=(0.0, 1000.0),
        sigma=(10.0, 13.0),
    )
    _, p0, p1, p2 = np.nonzero(seg_result)
    seg_points = np.concatenate([p0, p1, p2])[None, None]
    assert np.allclose(seg_points, np.round(points_result))


@patch("nndet.io.transforms.random.spatial.np.random.uniform", Mock(return_value=0.5))
def test_scale_spatial_fn(example):
    data, seg, points = example
    _, seg_result, points_result = augment_spatial(
        data=data,
        seg=seg,
        points=points,
        do_elastic_deform=False,
        do_rotation=False,
        do_scale=True,
        scale=(0.5, 0.5),
        p_scale_per_sample=1.0,
        patch_size=(12, 12, 12),
    )
    d = 7.0 + (9.0 - 7.0) * 1 / 0.5 + 3.0 / 2.0  # 7.0 + (9.0 - 7.0) * 1 / 0.5 for p=15
    expected_points = np.array([[[d, d, d]]], dtype=float)
    assert np.allclose(expected_points, points_result)


def test_rot90_x_spatial_fn(example):
    data, seg, points = example
    _, seg_result, points_result = augment_spatial(
        data=data,
        seg=seg,
        points=points,
        do_elastic_deform=False,
        do_rotation=True,
        do_scale=False,
        patch_size=(12, 12, 12),
        angle_x=(np.pi / 2, np.pi / 2),
        angle_y=(0, 0),
        angle_z=(0, 0),
        p_rot_per_sample=1.0,
    )
    expected_points = np.array([[[9.0, 2.0, 9.0]]], dtype=float)
    assert np.allclose(expected_points, points_result)
    _, p0, p1, p2 = np.nonzero(seg_result)
    seg_points = np.concatenate([p0, p1, p2])[None, None]
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
        patch_size=(12, 12, 12),
        angle_x=(0, 0),
        angle_y=(np.pi / 2, np.pi / 2),
        angle_z=(0, 0),
        p_rot_per_sample=1.0,
    )
    expected_points = np.array([[[9.0, 9.0, 2.0]]], dtype=float)
    assert np.allclose(expected_points, points_result)
    _, p0, p1, p2 = np.nonzero(seg_result)
    seg_points = np.concatenate([p0, p1, p2])[None, None]
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
        patch_size=(12, 12, 12),
        angle_x=(0, 0),
        angle_y=(0, 0),
        angle_z=(np.pi / 2, np.pi / 2),
        p_rot_per_sample=1.0,
    )
    expected_points = np.array([[[2.0, 9.0, 9.0]]], dtype=float)
    assert np.allclose(expected_points, points_result)
    _, p0, p1, p2 = np.nonzero(seg_result)
    seg_points = np.concatenate([p0, p1, p2])[None, None]
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
    seg_points = np.concatenate([p0, p1, p2])[None, None]
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
    seg_points = np.concatenate([p0, p1, p2])[None, None]
    assert np.allclose(seg_points, points_result)


def test_spatial_nndet_bg():
    # test against bg spatial augment
    for i in range(1):
        np.random.seed(i + 100)
        data = np.random.rand(1, 1, 15, 15, 15)
        seg = np.zeros((1, 1, 15, 15, 15))
        seg[0, 0, 7, 7, 7] = 1
        seg[0, 0, 12, 14, 7] = 2
        seg[0, 0, 2:4, 4:8, 1:5] = 3
        p = 15
        # p=16 gives error because batchgen does not renorm with patch size

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
        np.random.seed(i)
        data_result, seg_result, _ = augment_spatial(
            data=data[0],
            seg=seg[0],
            do_elastic_deform=True,
            do_rotation=True,
            do_scale=True,
            patch_size=(p, p, p),
        )
        assert np.allclose(data_result_bg[0], data_result)
        assert np.allclose(seg_result_bg[0], seg_result)
