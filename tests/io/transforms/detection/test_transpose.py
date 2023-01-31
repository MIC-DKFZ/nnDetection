from unittest.mock import patch

import numpy as np
import pytest

import nndet.core.ops_np as ops_np
from nndet.io.transforms.detection.transpose import (
    TransposeAxesTransform,
    transpose_array,
    transpose_points,
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
def transpose_matrix():
    return np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


@pytest.fixture
def points_transposed():
    return np.array(
        [
            [-1, -1, -1, 1],
            [-1, -1, 1, 1],
            [1, -1, -1, 1],
            [1, -1, 1, 1],
            [-1, 1, -1, 1],
            [-1, 1, 1, 1],
            [1, 1, -1, 1],
            [1, 1, 1, 1],
        ],
        dtype=float,
    )  # [1, 8, 3 + 1]


def test_transpose_array():
    a = np.random.rand(1, 16, 16, 16)
    axes = (1, 0, 2)
    expected_a = a.transpose((0, 2, 1, 3))
    produced_a = transpose_array(a, axes=axes)
    assert np.allclose(produced_a, expected_a)


def test_transpose_points(points_origin, transpose_matrix, points_transposed):
    produced_points = transpose_points(points_origin, matrix=transpose_matrix)
    assert np.allclose(produced_points, points_transposed)

    # 2x mirror should yield original points
    oiriginal_points = transpose_points(produced_points, matrix=transpose_matrix)
    assert np.allclose(oiriginal_points, points_origin)


def test_get_axes():
    np.random.seed(0)
    produced_axes = TransposeAxesTransform.get_axes(axes=(0, 1), ndim=3)
    expected_axes = [1, 0, 2]
    assert produced_axes == expected_axes


def test_get_matrix(transpose_matrix):
    produced_matrix = TransposeAxesTransform.get_matrix(transpose_axes=[1, 0, 2], ndim=3)
    assert np.allclose(produced_matrix, transpose_matrix)


@patch("nndet.io.transforms.detection.transpose.np.random.uniform", lambda: 0.4)
def test_transpose_transform(points_origin, points_transposed):
    np.random.seed(7)
    batch = {
        "data": np.random.rand(2, 1, 16, 16, 16),
        "seg": np.random.rand(2, 1, 16, 16, 16),
        "point": [points_origin, points_origin],
    }

    expecetd_data = np.copy(batch["data"]).transpose((0, 1, 3, 2, 4))
    expecetd_seg = np.copy(batch["seg"]).transpose((0, 1, 3, 2, 4))
    expected_point = [points_transposed, points_transposed]

    trafo = TransposeAxesTransform(data_key="data", label_key="seg", point_key="point", axes=(0, 1))
    batch_produced = trafo(**batch)

    assert np.allclose(batch_produced["data"], expecetd_data)
    assert np.allclose(batch_produced["seg"], expecetd_seg)
    assert len(batch_produced["point"]) == len(expected_point)
    for bp, ep in zip(batch_produced["point"], expected_point):
        assert np.allclose(bp, ep)


@patch("nndet.io.transforms.detection.mirror.np.random.uniform", lambda: 0.4)
def test_transpose_points_seg():
    for coords in [(0, 0, 0, 0), (0, 4, 5, 8), (0, 15, 15, 15), (0, 4, slice(3, 6), slice(8, 12))]:
        seg = np.zeros((1, 16, 16, 16))
        img_shape = (16, 16, 16)
        seg[coords] = 1

        boxes = instances_to_boxes_np(seg, dim=3)[0]
        input_points = ops_np.points_to_homogeneous([ops_np.boxes2corner_points(boxes).astype(float)])[0]

        for axes in [(0, 1, 2), (1, 0, 2), (2, 1, 0), (0, 2, 1), (2, 0, 1)]:
            seg_new = transpose_array(seg, axes=axes)
            boxes_new = instances_to_boxes_np(seg_new, dim=3)[0]
            seg_new_points = ops_np.boxes2corner_points(boxes_new).astype(float)[0]

            matrix = TransposeAxesTransform.get_matrix(transpose_axes=axes, ndim=len(img_shape))
            produced_points = transpose_points(input_points, matrix=matrix)[0, :, :3]

            assert produced_points.shape == seg_new_points.shape
            num_points = produced_points.shape[0]
            for i in range(num_points):
                assert np.prod(
                    np.isclose(produced_points[i][None].repeat(num_points, axis=0), seg_new_points), axis=1
                ).any()
