import numpy as np

import nndet.core.ops_np as ops_np

## test point ops ##


def test_boxes2corner_points_2d():
    boxes = np.array([[0, 0, 1, 1]])
    points = ops_np.boxes2corner_points(boxes)
    expected_points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    assert np.allclose(expected_points, points)


def test_boxes2corner_points_3d():
    boxes = np.array([[0, 0, 1, 1, 0, 1]])
    points = ops_np.boxes2corner_points(boxes)
    expected_points = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    assert np.allclose(
        expected_points,
        points,
    )


def test_boxes2center_area_points():
    boxes = np.array([[0, 0, 1, 1, 0, 1]])
    points = ops_np.boxes2center_area_points(boxes)
    expected_points = np.array(
        [
            [1, 0.5, 0.5],
            [0, 0.5, 0.5],
            [0.5, 1, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 1.0],
            [0.5, 0.5, 0],
        ]
    )
    assert np.allclose(points, expected_points)


def test_polygon_points2boxes_0():
    # these ppoints are tested in test_boxes2corner_points_3d
    expected_boxes = np.array([[0, 0, 1, 1, 0, 1]])
    points = ops_np.boxes2corner_points(expected_boxes)
    boxes = ops_np.object_points2boxes(points)
    assert np.allclose(expected_boxes, boxes)


def test_polygon_points2boxes_1():
    # these ppoints are tested in test_boxes2center_area_points
    expected_boxes = np.array([[0, 0, 1, 1, 0, 1]])
    points = ops_np.boxes2center_area_points(expected_boxes)
    boxes = ops_np.object_points2boxes(points)
    assert np.allclose(expected_boxes, boxes)
