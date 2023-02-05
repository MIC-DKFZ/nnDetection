from unittest.mock import patch

import numpy as np
import pytest

import nndet.core.ops_np as ops_np
from nndet.io.transforms.detection.mirror import (
    MirrorTransform,
    mirror_array,
    mirror_points,
)
from nndet.io.transforms.instances import instances_to_boxes_np


@pytest.fixture
def points_origin():
    return np.array(
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


@pytest.fixture
def mirror_all_16():
    return np.array(
        [
            [-1, 0, 0, 15],
            [0, -1, 0, 15],
            [0, 0, -1, 15],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


@pytest.fixture
def points_mirrored_all_16():
    return np.array(
        [
            [16, 16, 16, 1],
            [16, 16, 14, 1],
            [16, 14, 16, 1],
            [16, 14, 14, 1],
            [14, 16, 16, 1],
            [14, 16, 14, 1],
            [14, 14, 16, 1],
            [14, 14, 14, 1],
        ],
        dtype=float,
    )


def test_mirror_array():
    a = np.random.rand(1, 16, 16, 16)
    axes = (0, 1, 2)
    expected_a = a[:, ::-1, ::-1, ::-1]
    produced_a = mirror_array(a, axes=axes)
    assert np.allclose(produced_a, expected_a)


def test_mirror_points(points_origin, mirror_all_16, points_mirrored_all_16):
    points = points_origin
    matrix = mirror_all_16
    expecetd_points = points_mirrored_all_16
    produced_points = mirror_points(points, matrix=matrix)
    assert np.allclose(produced_points, expecetd_points)

    # 2x mirror should yield original points
    oiriginal_points = mirror_points(produced_points, matrix=matrix)
    assert np.allclose(oiriginal_points, points)


def test_get_matrix_all():
    # all axes
    produced_matrix = MirrorTransform.get_matrix((0, 1, 2), img_shape=(16, 16, 16))
    expected_matrix = np.array(
        [
            [-1, 0, 0, 15],
            [0, -1, 0, 15],
            [0, 0, -1, 15],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    assert np.allclose(produced_matrix, expected_matrix)


def test_get_matrix_first():
    # first axis
    produced_matrix = MirrorTransform.get_matrix((0,), img_shape=(16, 16, 16))
    expected_matrix = np.array(
        [
            [-1, 0, 0, 15],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    assert np.allclose(produced_matrix, expected_matrix)


@patch("nndet.io.transforms.detection.mirror.np.random.uniform", lambda: 0.6)
def test_get_axes_empty():
    axes = MirrorTransform.get_axes(axes=(0, 1, 2))
    assert not axes


@patch("nndet.io.transforms.detection.mirror.np.random.uniform", lambda: 0.4)
def test_get_axes_all():
    axes = MirrorTransform.get_axes(axes=(0, 1, 2))
    assert axes == [0, 1, 2]

    axes = MirrorTransform.get_axes(axes=(1, 2))
    assert axes == [1, 2]


@patch("nndet.io.transforms.detection.mirror.np.random.uniform", lambda: 0.4)
def test_mirror_transform(points_origin, points_mirrored_all_16):
    batch = {
        "data": np.random.rand(2, 1, 16, 16, 16),
        "seg": np.random.rand(2, 1, 16, 16, 16),
        "point": [points_origin, points_origin],
    }

    expecetd_data = np.copy(batch["data"])[:, :, ::-1, ::-1, ::-1]
    expecetd_seg = np.copy(batch["seg"])[:, :, ::-1, ::-1, ::-1]
    expected_point = [points_mirrored_all_16, points_mirrored_all_16]

    trafo = MirrorTransform(data_key="data", label_key="seg", point_key="point")
    batch_produced = trafo(**batch)

    assert np.allclose(batch_produced["data"], expecetd_data)
    assert np.allclose(batch_produced["seg"], expecetd_seg)
    assert len(batch_produced["point"]) == len(expected_point)
    for bp, ep in zip(batch_produced["point"], expected_point):
        assert np.allclose(bp, ep)


@patch("nndet.io.transforms.detection.mirror.np.random.uniform", lambda: 0.4)
def test_mirror_points_seg():
    for coords in [(0, 0, 0, 0), (0, 4, 5, 8), (0, 15, 15, 15), (0, 4, slice(3, 6), slice(8, 12))]:
        seg = np.zeros((1, 16, 16, 16))
        img_shape = (16, 16, 16)
        seg[coords] = 1

        boxes = instances_to_boxes_np(seg, dim=3)[0]
        input_points = ops_np.points_to_homogeneous([ops_np.boxes2corner_points(boxes).astype(float)])[0]

        for axes in [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]:
            seg_new = mirror_array(seg, axes=axes)
            boxes_new = instances_to_boxes_np(seg_new, dim=3)[0]
            seg_new_points = ops_np.boxes2corner_points(boxes_new).astype(float)[0]

            matrix = MirrorTransform.get_matrix(axes=axes, img_shape=img_shape)
            produced_points = mirror_points(input_points, matrix=matrix)[0, :, :3]

            assert produced_points.shape == seg_new_points.shape
            num_points = produced_points.shape[0]
            for i in range(num_points):
                assert np.prod(
                    np.isclose(produced_points[i][None].repeat(num_points, axis=0), seg_new_points), axis=1
                ).any()
