import numpy as np
import pytest

import nndet.core.ops_np as ops_np


@pytest.fixture
def boxes0_2d():
    return np.array([[0, 0, 2, 2], [0, 0, 2, 2]]).astype(float)


@pytest.fixture
def boxes1_2d():
    return np.array([[1, 1, 3, 3], [1, 1, 3, 3], [1, 1, 3, 3]]).astype(float)


@pytest.fixture
def boxes2_2d():
    return np.array([[1, 1, 2, 2], [1, 1, 3, 3]]).astype(float)


@pytest.fixture
def boxes0_3d():
    return np.array([[0, 0, 2, 2, 0, 2], [0, 0, 2, 2, 0, 2]]).astype(float)


@pytest.fixture
def boxes1_3d():
    return np.array([[1, 1, 3, 3, 1, 3], [1, 1, 3, 3, 1, 3], [1, 1, 3, 3, 1, 3]]).astype(float)


@pytest.fixture
def boxes2_3d():
    return np.array([[1, 1, 2, 2, 1, 2], [1, 1, 3, 3, 1, 3], [1, 1, 4, 4, 1, 4]]).astype(float)


def test_box_size_2d(boxes0_2d, boxes1_2d):
    size0 = ops_np.box_size_np(boxes0_2d)
    size1 = ops_np.box_size_np(boxes1_2d)

    assert np.allclose(size0, [[2, 2], [2, 2]])
    assert np.allclose(size1, [[2, 2], [2, 2], [2, 2]])


def test_box_size_3d(boxes0_3d, boxes1_3d):
    size0 = ops_np.box_size_np(boxes0_3d)
    size1 = ops_np.box_size_np(boxes1_3d)

    assert np.allclose(size0, [[2, 2, 2], [2, 2, 2]])
    assert np.allclose(size1, [[2, 2, 2], [2, 2, 2], [2, 2, 2]])


def test_remove_small_boxes_2d(boxes2_2d):
    idx_b0_s0 = ops_np.remove_small_boxes(boxes2_2d, min_size=0)
    assert np.allclose(idx_b0_s0, np.array([0, 1], dtype=np.int64))

    idx_b0_s0 = ops_np.remove_small_boxes(boxes2_2d, min_size=2)
    assert np.allclose(idx_b0_s0, np.array([1], dtype=np.int64))

    idx_b0_s0 = ops_np.remove_small_boxes(boxes2_2d, min_size=3)
    assert np.allclose(idx_b0_s0, np.array([], dtype=np.int64))


def test_remove_small_boxes_3d(boxes2_3d):
    idx_b0_s0 = ops_np.remove_small_boxes(boxes2_3d, min_size=0)
    assert np.allclose(idx_b0_s0, np.array([0, 1, 2], dtype=np.int64))

    idx_b0_s0 = ops_np.remove_small_boxes(boxes2_3d, min_size=2)
    assert np.allclose(idx_b0_s0, np.array([1, 2], dtype=np.int64))

    idx_b0_s0 = ops_np.remove_small_boxes(boxes2_3d, min_size=4)
    assert np.allclose(idx_b0_s0, np.array([], dtype=np.int64))


def test_clip_boxes_2d():
    boxes = np.array([[-1, -2, 102, 200]])
    boxes = ops_np.clip_boxes_to_image(boxes, (100, 200))
    expected = np.array([[-1, -1, 100, 200]])
    assert (expected == boxes).all()


def test_clip_boxes_3d():
    boxes = np.array([[-1, -2, 102, 200, -5, 30]])
    boxes = ops_np.clip_boxes_to_image(boxes, (100, 200, 16))
    expected = np.array([[-1, -1, 100, 200, -1, 16]])
    assert (expected == boxes).all()
