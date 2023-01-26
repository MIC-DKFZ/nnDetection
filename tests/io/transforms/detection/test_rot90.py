from unittest.mock import patch

import numpy as np
import pytest

import nndet.core.ops_np as ops_np
from nndet.io.transforms.instances import instances_to_boxes_np
from nndet.io.transforms.detection.rot90 import Rot90Transform, rot90_array, rot90_points


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


def test_rot90_array():
    a = np.random.rand(1, 16, 16, 16)
    axes = (0, 1)
    num_rot = 3
    expected_a = np.rot90(a, k=num_rot, axes=(1, 2))
    produced_a = rot90_array(a, num_rot=num_rot, axes=axes)
    assert np.allclose(produced_a, expected_a)


def test_rot90_points_seg(points_origin):
    seg = np.zeros((1, 16, 16, 16))
    img_shape = (16, 16, 16)
    seg[0, 0, 0, 0] = 1

    for axes in [(0, 1), (1, 2), (1, 0), (2, 1), (0, 2), (2, 0)]:
        for num_rot in range(1, 4):
            seg_rot = rot90_array(seg, num_rot=num_rot, axes=axes)
            boxes_rot = instances_to_boxes_np(seg_rot, dim=3)[0]
            seg_points = ops_np.boxes2corner_points(boxes_rot).astype(float)[0]

            matrix = Rot90Transform.get_matrix(axes=axes, num_rot=num_rot, img_shape=img_shape)
            produced_points = rot90_points(points_origin, matrix=matrix)[:, :3]

            assert produced_points.shape == seg_points.shape
            num_points = produced_points.shape[0]
            for i in range(num_points):
                assert np.prod(
                    np.isclose(produced_points[i][None].repeat(num_points, axis=0), seg_points), axis=1
                ).any()


def test_rot90_points_seg2():
    for coords in [(0, 0, 0, 0), (0, 4, 5, 8), (0, 15, 15, 15), (0, 4, slice(3, 6), slice(8, 12))]:
        seg = np.zeros((1, 16, 16, 16))
        img_shape = (16, 16, 16)
        seg[coords] = 1

        boxes = instances_to_boxes_np(seg, dim=3)[0]
        input_points = ops_np.points_to_homogeneous([ops_np.boxes2corner_points(boxes).astype(float)])[0]

        for axes in [(0, 1), (1, 2), (1, 0), (2, 1), (0, 2), (2, 0)]:
            for num_rot in range(1, 4):
                seg_rot = rot90_array(seg, num_rot=num_rot, axes=axes)
                boxes_rot = instances_to_boxes_np(seg_rot, dim=3)[0]
                seg_points = ops_np.boxes2corner_points(boxes_rot).astype(float)[0]

                matrix = Rot90Transform.get_matrix(axes=axes, num_rot=num_rot, img_shape=img_shape)
                produced_points = rot90_points(input_points, matrix=matrix)[0, :, :3]

                assert produced_points.shape == seg_points.shape
                num_points = produced_points.shape[0]
                for i in range(num_points):
                    assert np.prod(
                        np.isclose(produced_points[i][None].repeat(num_points, axis=0), seg_points), axis=1
                    ).any()


def choice_wrapper(return_value):
    def fn(axes, size=None, replace=None):
        if len(axes) == 1:
            return axes[0]
        return np.array(return_value)

    return fn


def test_rot90_transform(points_origin):
    for axes in [(0, 1), (1, 2), (1, 0), (2, 1), (0, 2), (2, 0)]:
        for num_rot in range(1, 4):
            batch = {
                "data": np.random.rand(1, 1, 16, 16, 16),
                "seg": np.zeros((1, 1, 16, 16, 16)),
                "point": [points_origin[None]],
            }
            batch["seg"][0, 0, 0, 0, 0] = 1

            trafo = Rot90Transform(data_key="data", label_key="seg", point_key="point", num_rot=(num_rot,))
            with patch("nndet.io.transforms.detection.rot90.np.random.choice", choice_wrapper(axes)):
                batch_produced = trafo(**batch)

            boxes_rot = instances_to_boxes_np(batch_produced["seg"], dim=3)[0]
            seg_points = ops_np.boxes2corner_points(boxes_rot).astype(float)[0]
            produced_points = batch_produced["point"][0][0, :, :3]

            assert produced_points.shape == seg_points.shape
            num_points = produced_points.shape[0]
            for i in range(num_points):
                assert np.prod(
                    np.isclose(produced_points[i][None].repeat(num_points, axis=0), seg_points), axis=1
                ).any()
