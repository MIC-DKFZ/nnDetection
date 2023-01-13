from unittest.mock import patch

import numpy as np
import pytest

from nndet.io.datamodule.loader import BaseDataLoader3D
from nndet.io.patching import save_get_crop
from nndet.io.transforms.instances import instances_to_boxes_np


class DummyLoader:
    def __init__(self) -> None:
        self._data = {"c1": {"label_boxes_file": "boxes"}}
        self.patch_size_generator = (8, 8, 8)


@pytest.fixture
def example_case():
    data = np.zeros((1, 16, 16, 16))
    seg = np.zeros((16, 16, 16))
    seg[4:7, 4:7, 4:7] = 1
    seg[12:16, 0:4, 12:16] = 2
    seg[8:12, 8:12, 8:12] = 3
    return data, seg


EXAMPLE_CASE_EMPTY = {"boxes": np.array([[]]), "labels": np.array([])}
EXAMPLE_CASE = {
    "boxes": np.array(
        [
            [3, 3, 7, 7, 3, 7],
            [11, -1, 16, 4, 11, 16],
            [7, 7, 12, 12, 7, 12],
        ],
        dtype=float,
    ),
    "labels": np.array([1, 0, 1], dtype=int),
}
EXAMPLE_CASE_SINGLE_PIXEL = {
    "boxes": np.array(
        [
            [8, 8, 10, 10, 8, 10],
        ],
        dtype=float,
    ),
    "labels": np.array([2], dtype=int),
}


@patch("nndet.io.datamodule.loader.np.load", lambda x: EXAMPLE_CASE_EMPTY)
def test_load_box_from_crop_empty():
    io_coords, io_labels = BaseDataLoader3D.load_box_from_crop(
        DummyLoader(),
        case_id="c1",
        case_data=np.zeros((1, 16, 16, 16)),
        crop=None,
    )

    expected_coords = np.array([[]], dtype=float).reshape(-1, 3 * 2)
    expected_labels = np.array([], dtype=int)

    assert np.allclose(io_coords, expected_coords)
    assert np.allclose(io_labels, expected_labels)


@patch("nndet.io.datamodule.loader.np.load", lambda x: EXAMPLE_CASE)
def test_load_box_from_crop_obj1(example_case):
    crop = (slice(0, 8), slice(0, 8), slice(0, 8))
    case_data, case_seg = example_case
    io_coords, io_labels = BaseDataLoader3D.load_box_from_crop(
        DummyLoader(),
        case_id="c1",
        case_data=np.zeros((1, 16, 16, 16)),
        crop=crop,
    )

    # manual result
    expected_coords = np.array([[3, 3, 7, 7, 3, 7]], dtype=float)
    expected_labels = np.array([1], dtype=int)

    assert np.allclose(io_coords, expected_coords)
    assert np.allclose(io_labels, expected_labels)

    # io result from seg
    io_seg = save_get_crop(
        data=case_seg,
        crop=crop,
    )[0]
    seg_coords = instances_to_boxes_np(io_seg, dim=3)[0]
    assert np.allclose(seg_coords, io_coords)


@patch("nndet.io.datamodule.loader.np.load", lambda x: EXAMPLE_CASE)
def test_load_box_from_crop_obj2_crop1(example_case):
    crop = (slice(8, 16), slice(0, 8), slice(8, 16))
    case_data, case_seg = example_case
    io_coords, io_labels = BaseDataLoader3D.load_box_from_crop(
        DummyLoader(),
        case_id="c1",
        case_data=np.zeros((1, 16, 16, 16)),
        crop=[slice(8, 16), slice(0, 8), slice(8, 16)],
    )

    # manual result
    expected_coords = np.array([[3, -1, 8, 4, 3, 8]], dtype=float)
    expected_labels = np.array([0], dtype=int)

    assert np.allclose(io_coords, expected_coords)
    assert np.allclose(io_labels, expected_labels)

    # io result from seg
    io_seg = save_get_crop(
        data=case_seg,
        crop=crop,
    )[0]
    seg_coords = instances_to_boxes_np(io_seg, dim=3)[0]
    assert np.allclose(seg_coords, io_coords)


@patch("nndet.io.datamodule.loader.np.load", lambda x: EXAMPLE_CASE)
def test_load_box_from_crop_obj2_crop2(example_case):
    crop = (slice(8, 16), slice(2, 10), slice(6, 14))
    case_data, case_seg = example_case
    io_coords, io_labels = BaseDataLoader3D.load_box_from_crop(
        DummyLoader(),
        case_id="c1",
        case_data=np.zeros((1, 16, 16, 16)),
        crop=crop,
    )

    # manual result
    expected_coords = np.array(
        [
            [3, -1, 8, 2, 5, 8],
            [-1, 5, 4, 8, 1, 6],
        ],
        dtype=float,
    )
    expected_labels = np.array([0, 1], dtype=int)

    assert np.allclose(io_coords, expected_coords)
    assert np.allclose(io_labels, expected_labels)

    # io result from seg
    io_seg = save_get_crop(
        data=case_seg,
        crop=crop,
    )[0]
    seg_coords = instances_to_boxes_np(io_seg, dim=3)[0]
    assert np.allclose(seg_coords, io_coords)


@patch("nndet.io.datamodule.loader.np.load", lambda x: EXAMPLE_CASE_SINGLE_PIXEL)
def test_load_box_from_crop_single_pixel_cut():
    loader = DummyLoader()
    loader.patch_size_generator = (1, 1, 1)
    crop = (slice(9, 10), slice(9, 10), slice(9, 10))
    io_coords, io_labels = BaseDataLoader3D.load_box_from_crop(
        loader,
        case_id="c1",
        case_data=np.zeros((1, 16, 16, 16)),
        crop=crop,
    )

    expected_coords = np.array([[-1, -1, 1, 1, -1, 1]], dtype=float).reshape(-1, 3 * 2)
    expected_labels = np.array([2], dtype=int)

    assert np.allclose(io_coords, expected_coords)
    assert np.allclose(io_labels, expected_labels)


@patch("nndet.io.datamodule.loader.np.load", lambda x: EXAMPLE_CASE_EMPTY)
def test_load_box_from_crop_single_pixel_obj():
    crop = (slice(4, 12), slice(4, 12), slice(4, 12))
    io_coords, io_labels = BaseDataLoader3D.load_box_from_crop(
        DummyLoader(),
        case_id="c1",
        case_data=np.zeros((1, 16, 16, 16)),
        crop=crop,
    )

    expected_coords = np.array([[4, 4, 6, 6, 4, 6]], dtype=float).reshape(-1, 3 * 2)
    expected_labels = np.array([2], dtype=int)

    assert np.allclose(io_coords, expected_coords)
    assert np.allclose(io_labels, expected_labels)


# TODO: outside crop lower and upper bound
# TODO: test with different save_get modi
# TODO: recheck why seg needs to be padded with -1
# TODO: remove padding option for data -> constant pad 0
