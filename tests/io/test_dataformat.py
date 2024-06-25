import os

import numpy as np
import pytest

from nndet.io.dataformat import data_format_to_class_mapping


@pytest.fixture
def np_data():
    np.random.seed(141)
    return np.random.rand(1, 256, 256, 256)


@pytest.fixture
def np_seg():
    np.random.seed(141)
    return np.random.rand(1, 256, 256, 256)


def test_save_npz(np_data, np_seg):
    with_npz = data_format_to_class_mapping["npz"]
    with_npz.save("/tmp/case_0", data=np_data, seg=np_seg)
    assert os.path.isfile("/tmp/case_0.npz")


def test_load_npz(np_data, np_seg):
    with_npz = data_format_to_class_mapping["npz"]
    loaded_data = with_npz.load_data("/tmp/case_0.npz")
    loaded_seg = with_npz.load_seg("/tmp/case_0.npz")
    assert (loaded_data == np_data).all()
    assert (loaded_seg == np_seg).all()


def test_save_b2nd(np_data, np_seg):
    with_b2nd = data_format_to_class_mapping["b2nd"]
    with_b2nd.save("/tmp/case_0", data=np_data, seg=np_seg, patch_size=(128, 128, 128))
    assert os.path.isfile("/tmp/case_0.b2nd")
    assert os.path.isfile("/tmp/case_0_seg.b2nd")


def test_load_b2nd(np_data, np_seg):
    with_b2nd = data_format_to_class_mapping["b2nd"]
    loaded_data = with_b2nd.load_data("/tmp/case_0.b2nd")
    loaded_seg = with_b2nd.load_seg("/tmp/case_0_seg.b2nd")
    assert (loaded_data == np_data).all()
    assert (loaded_seg == np_seg).all()
