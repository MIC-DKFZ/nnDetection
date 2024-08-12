import numpy as np
import pytest
import torch

from nndet.preprocessing.torchresampling import resample_data_or_seg

np.random.seed(141)
torch.manual_seed(141)

npfloat32data = np.random.rand(1, 128, 256, 256).astype(np.float32)
npfloat16data = np.random.rand(2, 96, 320, 256).astype(np.float16)
npint8seg = np.random.randint(low=2**7 - 10, high=2**7, size=(1, 131, 256, 512)).astype(np.int8)
npint16seg = np.random.randint(low=2**15 - 10, high=2**15, size=(3, 115, 229, 256)).astype(np.int16)
ptfloat32data = torch.rand(1, 128, 256, 256).type(torch.float32)
ptfloat16data = torch.rand(2, 96, 320, 256).type(torch.float16)
ptint8seg = torch.randint(low=2**7 - 10, high=2**7, size=(1, 131, 256, 512)).type(torch.int8)
ptint16seg = torch.randint(low=2**15 - 10, high=2**15, size=(3, 115, 229, 256)).type(torch.int16)


@pytest.mark.parametrize(
    "data,new_shape,is_seg,memefficient_seg_resampling,mode,do_separate_z,axis",
    [
        # numpy float32 with linear interpolation
        (npfloat32data, (100, 196, 196), False, False, "linear", False, None),
        (npfloat32data, (51, 131, 155), False, False, "linear", False, None),
        # numpy float16 with linear interpolation
        (npfloat16data, (100, 196, 128), False, False, "linear", False, None),
        (npfloat16data, (51, 100, 155), False, False, "linear", False, None),
        # numpy seg int8 with linear interpolation
        (npint8seg, (100, 128, 256), True, False, "linear", False, None),
        (npint8seg, (51, 131, 155), True, False, "linear", False, None),
        # numpy seg int16 with linear interpolation
        (npint16seg, (75, 100, 100), True, False, "linear", False, None),
        (npint16seg, (51, 131, 155), True, False, "linear", False, None),
        # numpy seg int8 with linear interpolation with memefficient_seg_resampling
        (npint8seg, (100, 128, 256), True, True, "linear", False, None),
        (npint8seg, (51, 131, 155), True, True, "linear", False, None),
        # numpy seg int16 with linear interpolation with memefficient_seg_resampling
        (npint16seg, (75, 100, 100), True, True, "linear", False, None),
        (npint16seg, (51, 131, 155), True, True, "linear", False, None),
        # numpy seg int8 with linear interpolation
        (npint8seg, (100, 128, 256), True, False, "nearest", False, None),
        (npint8seg, (51, 131, 155), True, False, "nearest", False, None),
        # numpy seg int16 with linear interpolation
        (npint16seg, (75, 100, 100), True, False, "nearest", False, None),
        (npint16seg, (51, 131, 155), True, False, "nearest", False, None),
        # numpy seg int8 with linear interpolation with memefficient_seg_resampling
        (npint8seg, (100, 128, 256), True, True, "nearest", False, None),
        (npint8seg, (51, 131, 155), True, True, "nearest", False, None),
        # numpy seg int16 with linear interpolation with memefficient_seg_resampling
        (npint16seg, (75, 100, 100), True, True, "nearest", False, None),
        (npint16seg, (51, 131, 155), True, True, "nearest", False, None),
        # ----------------
        # pytorch float32 with linear interpolation
        (ptfloat32data, (100, 196, 196), False, False, "linear", False, None),
        (ptfloat32data, (51, 131, 155), False, False, "linear", False, None),
        # pytorch float16 with linear interpolation
        (ptfloat16data, (100, 196, 128), False, False, "linear", False, None),
        (ptfloat16data, (51, 100, 155), False, False, "linear", False, None),
        # pytorch seg int8 with linear interpolation
        (ptint8seg, (100, 128, 256), True, False, "linear", False, None),
        (ptint8seg, (51, 131, 155), True, False, "linear", False, None),
        # pytorch seg int16 with linear interpolation
        (ptint16seg, (75, 100, 100), True, False, "linear", False, None),
        (ptint16seg, (51, 131, 155), True, False, "linear", False, None),
        # pytorch seg int8 with linear interpolation with memefficient_seg_resampling
        (ptint8seg, (100, 128, 256), True, True, "linear", False, None),
        (ptint8seg, (51, 131, 155), True, True, "linear", False, None),
        # pytorch seg int16 with linear interpolation with memefficient_seg_resampling
        (ptint16seg, (75, 100, 100), True, True, "linear", False, None),
        (ptint16seg, (51, 131, 155), True, True, "linear", False, None),
        # pytorch seg int8 with linear interpolation
        (ptint8seg, (100, 128, 256), True, False, "nearest", False, None),
        (ptint8seg, (51, 131, 155), True, False, "nearest", False, None),
        # pytorch seg int16 with linear interpolation
        (ptint16seg, (75, 100, 100), True, False, "nearest", False, None),
        (ptint16seg, (51, 131, 155), True, False, "nearest", False, None),
        # pytorch seg int8 with linear interpolation with memefficient_seg_resampling
        (ptint8seg, (100, 128, 256), True, True, "nearest", False, None),
        (ptint8seg, (51, 131, 155), True, True, "nearest", False, None),
        # pytorch seg int16 with linear interpolation with memefficient_seg_resampling
        (ptint16seg, (75, 100, 100), True, True, "nearest", False, None),
        (ptint16seg, (51, 131, 155), True, True, "nearest", False, None),
        # make sure the it works for do_separate_z
        (npfloat32data, (100, 196, 196), False, False, "linear", True, [0]),
        (npfloat32data, (51, 131, 155), False, False, "linear", True, [1]),
        (npfloat16data, (100, 196, 128), False, False, "linear", True, [2]),
        (npfloat16data, (51, 100, 155), False, False, "linear", True, [0]),
        (npint8seg, (100, 128, 256), True, False, "linear", True, [1]),
        (npint8seg, (51, 131, 155), True, False, "linear", True, [2]),
        (npint16seg, (75, 100, 100), True, False, "linear", True, [0]),
        (npint16seg, (51, 131, 155), True, False, "linear", True, [1]),
        (npint8seg, (100, 128, 256), True, True, "linear", True, [2]),
        (npint8seg, (51, 131, 155), True, True, "linear", True, [0]),
        (npint16seg, (75, 100, 100), True, True, "linear", True, [1]),
        (npint16seg, (51, 131, 155), True, True, "linear", True, [2]),
        (npint8seg, (100, 128, 256), True, False, "nearest", True, [0]),
        (npint8seg, (51, 131, 155), True, False, "nearest", True, [1]),
        (npint16seg, (75, 100, 100), True, False, "nearest", True, [2]),
        (npint16seg, (51, 131, 155), True, False, "nearest", True, [0]),
        (npint8seg, (100, 128, 256), True, True, "nearest", True, [1]),
        (npint8seg, (51, 131, 155), True, True, "nearest", True, [2]),
        (npint16seg, (75, 100, 100), True, True, "nearest", True, [0]),
        (npint16seg, (51, 131, 155), True, True, "nearest", True, [1]),
        (ptfloat32data, (100, 196, 196), False, False, "linear", True, [1]),
        (ptfloat32data, (51, 131, 155), False, False, "linear", True, [2]),
        (ptfloat16data, (100, 196, 128), False, False, "linear", True, [0]),
        (ptfloat16data, (51, 100, 155), False, False, "linear", True, [1]),
        (ptint8seg, (100, 128, 256), True, False, "linear", True, [2]),
        (ptint8seg, (51, 131, 155), True, False, "linear", True, [0]),
        (ptint16seg, (75, 100, 100), True, False, "linear", True, [1]),
        (ptint16seg, (51, 131, 155), True, False, "linear", True, [2]),
        (ptint8seg, (100, 128, 256), True, True, "linear", True, [0]),
        (ptint8seg, (51, 131, 155), True, True, "linear", True, [1]),
        (ptint16seg, (75, 100, 100), True, True, "linear", True, [2]),
        (ptint16seg, (51, 131, 155), True, True, "linear", True, [0]),
        (ptint8seg, (100, 128, 256), True, False, "nearest", True, [1]),
        (ptint8seg, (51, 131, 155), True, False, "nearest", True, [2]),
        (ptint16seg, (75, 100, 100), True, False, "nearest", True, [0]),
        (ptint16seg, (51, 131, 155), True, False, "nearest", True, [1]),
        (ptint8seg, (100, 128, 256), True, True, "nearest", True, [2]),
        (ptint8seg, (51, 131, 155), True, True, "nearest", True, [0]),
        (ptint16seg, (75, 100, 100), True, True, "nearest", True, [1]),
        (ptint16seg, (51, 131, 155), True, True, "nearest", True, [2]),
    ],
)
def test_resample_data_or_seg(data, new_shape, is_seg, memefficient_seg_resampling, mode, do_separate_z, axis):
    resampled_data = resample_data_or_seg(
        data,
        new_shape,
        is_seg,
        memefficient_seg_resampling=memefficient_seg_resampling,
        mode=mode,
        do_separate_z=do_separate_z,
        axis=axis,
    )
    assert resampled_data.shape[1:] == new_shape
    assert data.shape[0] == resampled_data.shape[0]
    assert data.dtype == resampled_data.dtype
