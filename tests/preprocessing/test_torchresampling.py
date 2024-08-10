import os
from pathlib import Path

import torch
import numpy as np
import pytest

from nndet.preprocessing.torchresampling import resample_data_or_seg


def npfloat32data():
    np.random.seed(141)
    return np.random.rand(1, 128, 256, 256).astype(np.float32)


def npfloat16data():
    np.random.seed(141)
    return np.random.rand(2, 96, 320, 256).astype(np.float16)


def npint8seg():
    np.random.seed(141)
    return np.random.randint(low=2**7 - 10, high=2**7, size=(1, 131, 256, 512)).astype(np.int8)


def npint16seg():
    np.random.seed(141)
    return np.random.randint(low=2**15 - 10, high=2**15, size=(3, 115, 229, 256)).astype(np.int16)


def ptfloat32data():
    torch.manual_seed(141)
    return torch.rand(1, 128, 256, 256).type(torch.float32)


def ptfloat16data():
    torch.manual_seed(141)
    return torch.rand(2, 96, 320, 256).type(torch.float16)


def ptint8seg():
    torch.manual_seed(141)
    return torch.randint(low=2**7 - 10, high=2**7, size=(1, 131, 256, 512)).type(torch.int8)


def ptint16seg():
    torch.manual_seed(141)
    return torch.randint(low=2**15 - 10, high=2**15, size=(3, 115, 229, 256)).type(torch.int16)


@pytest.mark.parametrize(
    "data,new_shape,is_seg,memefficient_seg_resampling,mode",
    [
        # numpy float32 with linear interpolation
        (npfloat32data(), (100, 196, 196), False, False, "linear"),
        (npfloat32data(), (51, 131, 155), False, False, "linear"),
        # numpy float16 with linear interpolation
        (npfloat16data(), (100, 196, 128), False, False, "linear"),
        (npfloat16data(), (51, 100, 155), False, False, "linear"),
        # numpy seg int8 with linear interpolation
        (npint8seg(), (100, 128, 256), True, False, "linear"),
        (npint8seg(), (51, 131, 155), True, False, "linear"),
        # numpy seg int16 with linear interpolation
        (npint16seg(), (75, 100, 100), True, False, "linear"),
        (npint16seg(), (51, 131, 155), True, False, "linear"),
        # numpy seg int8 with linear interpolation with memefficient_seg_resampling
        (npint8seg(), (100, 128, 256), True, True, "linear"),
        (npint8seg(), (51, 131, 155), True, True, "linear"),
        # numpy seg int16 with linear interpolation with memefficient_seg_resampling
        (npint16seg(), (75, 100, 100), True, True, "linear"),
        (npint16seg(), (51, 131, 155), True, True, "linear"),
        # numpy seg int8 with linear interpolation
        (npint8seg(), (100, 128, 256), True, False, "nearest"),
        (npint8seg(), (51, 131, 155), True, False, "nearest"),
        # numpy seg int16 with linear interpolation
        (npint16seg(), (75, 100, 100), True, False, "nearest"),
        (npint16seg(), (51, 131, 155), True, False, "nearest"),
        # numpy seg int8 with linear interpolation with memefficient_seg_resampling
        (npint8seg(), (100, 128, 256), True, True, "nearest"),
        (npint8seg(), (51, 131, 155), True, True, "nearest"),
        # numpy seg int16 with linear interpolation with memefficient_seg_resampling
        (npint16seg(), (75, 100, 100), True, True, "nearest"),
        (npint16seg(), (51, 131, 155), True, True, "nearest"),
        # ----------------
        # pytorch float32 with linear interpolation
        (ptfloat32data(), (100, 196, 196), False, False, "linear"),
        (ptfloat32data(), (51, 131, 155), False, False, "linear"),
        # pytorch float16 with linear interpolation
        (ptfloat16data(), (100, 196, 128), False, False, "linear"),
        (ptfloat16data(), (51, 100, 155), False, False, "linear"),
        # pytorch seg int8 with linear interpolation
        (ptint8seg(), (100, 128, 256), True, False, "linear"),
        (ptint8seg(), (51, 131, 155), True, False, "linear"),
        # pytorch seg int16 with linear interpolation
        (ptint16seg(), (75, 100, 100), True, False, "linear"),
        (ptint16seg(), (51, 131, 155), True, False, "linear"),
        # pytorch seg int8 with linear interpolation with memefficient_seg_resampling
        (ptint8seg(), (100, 128, 256), True, True, "linear"),
        (ptint8seg(), (51, 131, 155), True, True, "linear"),
        # pytorch seg int16 with linear interpolation with memefficient_seg_resampling
        (ptint16seg(), (75, 100, 100), True, True, "linear"),
        (ptint16seg(), (51, 131, 155), True, True, "linear"),
        # pytorch seg int8 with linear interpolation
        (ptint8seg(), (100, 128, 256), True, False, "nearest"),
        (ptint8seg(), (51, 131, 155), True, False, "nearest"),
        # pytorch seg int16 with linear interpolation
        (ptint16seg(), (75, 100, 100), True, False, "nearest"),
        (ptint16seg(), (51, 131, 155), True, False, "nearest"),
        # pytorch seg int8 with linear interpolation with memefficient_seg_resampling
        (ptint8seg(), (100, 128, 256), True, True, "nearest"),
        (ptint8seg(), (51, 131, 155), True, True, "nearest"),
        # pytorch seg int16 with linear interpolation with memefficient_seg_resampling
        (ptint16seg(), (75, 100, 100), True, True, "nearest"),
        (ptint16seg(), (51, 131, 155), True, True, "nearest"),
    ],
)
def test_resample_data_or_seg(data, new_shape, is_seg, memefficient_seg_resampling, mode):
    resampled_data = resample_data_or_seg(
        data, new_shape, is_seg, memefficient_seg_resampling=memefficient_seg_resampling, mode=mode
    )
    assert resampled_data.shape[1:] == new_shape
    assert data.shape[0] == resampled_data.shape[0]
    assert data.dtype == resampled_data.dtype
