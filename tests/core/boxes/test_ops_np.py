import numpy as np
import pytest

from nndet.core.boxes import *


@pytest.fixture
def boxes0_2d():
    return np.array([[0, 0, 2, 2], [0, 0, 2, 2]]).astype(float)


@pytest.fixture
def boxes1_2d():
    return np.array([[1, 1, 3, 3], [1, 1, 3, 3], [1, 1, 3, 3]]).astype(float)


@pytest.fixture
def boxes0_3d():
    return np.array([[0, 0, 2, 2, 0, 2], [0, 0, 2, 2, 0, 2]]).astype(float)


@pytest.fixture
def boxes1_3d():
    return np.array([[1, 1, 3, 3, 1, 3], [1, 1, 3, 3, 1, 3], [1, 1, 3, 3, 1, 3]]).astype(float)


def test_box_size_2d(boxes0_2d, boxes1_2d):
    size0 = box_size_np(boxes0_2d)
    size1 = box_size_np(boxes1_2d)

    assert np.allclose(size0, [[2, 2], [2, 2]])
    assert np.allclose(size1, [[2, 2], [2, 2], [2, 2]])


def test_box_size_3d(boxes0_3d, boxes1_3d):
    size0 = box_size_np(boxes0_3d)
    size1 = box_size_np(boxes1_3d)

    assert np.allclose(size0, [[2, 2, 2], [2, 2, 2]])
    assert np.allclose(size1, [[2, 2, 2], [2, 2, 2], [2, 2, 2]])
