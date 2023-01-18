from unittest.mock import patch

import numpy as np
import pytest

from nndet.io.transforms.random.mirror import (
    MirrorTransform,
    mirror_array,
    mirror_points,
)


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


@patch("nndet.io.transforms.random.mirror.np.random.uniform", lambda: 0.6)
def test_get_axes_empty():
    axes = MirrorTransform.get_axes(axes=(0, 1, 2))
    assert not axes


@patch("nndet.io.transforms.random.mirror.np.random.uniform", lambda: 0.4)
def test_get_axes_all():
    axes = MirrorTransform.get_axes(axes=(0, 1, 2))
    assert axes == [0, 1, 2]

    axes = MirrorTransform.get_axes(axes=(1, 2))
    assert axes == [1, 2]


@patch("nndet.io.transforms.random.mirror.np.random.uniform", lambda: 0.4)
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
