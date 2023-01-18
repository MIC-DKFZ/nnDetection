from unittest.mock import patch

import numpy as np
import pytest

from nndet.io.transforms.instances import instances_to_boxes_np
from nndet.io.transforms.random.rot90 import Rot90Transform, rot90_array, rot90_points


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
def rotation_matrix():
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
def points_rot():
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


# [x, o]
# [o, o]


def test_rot90_array():
    a = np.random.rand(1, 16, 16, 16)
    axes = (0, 1)
    num_rot = 3
    expected_a = np.rot90(a, k=num_rot, axes=(1, 2))
    produced_a = rot90_array(a, num_rot=num_rot, axes=axes)
    assert np.allclose(produced_a, expected_a)


def test_rot90_points():
    pass


def test_rot90_points_seg():
    seg = np.zeros((1, 16, 16, 16))
    seg[0, 0, 0, 0] = 1
    boxes = instances_to_boxes_np(seg, dim=3)[0]
    print(boxes)

    seg_rot = rot90_array(seg, num_rot=3, axes=(0, 1))
    boxes_rot = instances_to_boxes_np(seg_rot, dim=3)[0]
    print(boxes_rot)
    raise


def test_get_axes():
    pass


def test_get_matrix():
    pass


def test_rot90_transform():
    pass
