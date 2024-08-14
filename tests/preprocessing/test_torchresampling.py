import numpy as np
import pytest
import torch

from nndet.preprocessing.torchresampling import resample_data_or_seg_torch

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
    "data,new_shape,is_seg,memefficient_seg_resampling,mode,aniso_axis_mode,do_separate_z,axis",
    [
        # numpy float32 with trilinear interpolation
        (npfloat32data, (100, 196, 196), False, False, "trilinear", "nearest-exact", False, None),
        (npfloat32data, (51, 131, 155), False, False, "trilinear", "nearest-exact", False, None),
        # numpy float16 with trilinear interpolation
        (npfloat16data, (100, 196, 128), False, False, "trilinear", "nearest-exact", False, None),
        (npfloat16data, (51, 100, 155), False, False, "trilinear", "nearest-exact", False, None),
        # numpy seg int8 with trilinear interpolation
        (npint8seg, (100, 128, 256), True, False, "trilinear", "nearest-exact", False, None),
        (npint8seg, (51, 131, 155), True, False, "trilinear", "nearest-exact", False, None),
        # numpy seg int16 with trilinear interpolation
        (npint16seg, (75, 100, 100), True, False, "trilinear", "nearest-exact", False, None),
        (npint16seg, (51, 131, 155), True, False, "trilinear", "nearest-exact", False, None),
        # numpy seg int8 with trilinear interpolation with memefficient_seg_resampling
        (npint8seg, (100, 128, 256), True, True, "trilinear", "nearest-exact", False, None),
        (npint8seg, (51, 131, 155), True, True, "trilinear", "nearest-exact", False, None),
        # numpy seg int16 with trilinear interpolation with memefficient_seg_resampling
        (npint16seg, (75, 100, 100), True, True, "trilinear", "nearest-exact", False, None),
        (npint16seg, (51, 131, 155), True, True, "trilinear", "nearest-exact", False, None),
        # numpy seg int8 with trilinear interpolation
        (npint8seg, (100, 128, 256), True, False, "nearest", "nearest-exact", False, None),
        (npint8seg, (51, 131, 155), True, False, "nearest", "nearest-exact", False, None),
        # numpy seg int16 with trilinear interpolation
        (npint16seg, (75, 100, 100), True, False, "nearest", "nearest-exact", False, None),
        (npint16seg, (51, 131, 155), True, False, "nearest", "nearest-exact", False, None),
        # numpy seg int8 with trilinear interpolation with memefficient_seg_resampling
        (npint8seg, (100, 128, 256), True, True, "nearest", "nearest-exact", False, None),
        (npint8seg, (51, 131, 155), True, True, "nearest", "nearest-exact", False, None),
        # numpy seg int16 with trilinear interpolation with memefficient_seg_resampling
        (npint16seg, (75, 100, 100), True, True, "nearest", "nearest-exact", False, None),
        (npint16seg, (51, 131, 155), True, True, "nearest", "nearest-exact", False, None),
        # ----------------
        # pytorch float32 with trilinear interpolation
        (ptfloat32data, (100, 196, 196), False, False, "trilinear", "nearest-exact", False, None),
        (ptfloat32data, (51, 131, 155), False, False, "trilinear", "nearest-exact", False, None),
        # pytorch float16 with trilinear interpolation
        (ptfloat16data, (100, 196, 128), False, False, "trilinear", "nearest-exact", False, None),
        (ptfloat16data, (51, 100, 155), False, False, "trilinear", "nearest-exact", False, None),
        # pytorch seg int8 with trilinear interpolation
        (ptint8seg, (100, 128, 256), True, False, "trilinear", "nearest-exact", False, None),
        (ptint8seg, (51, 131, 155), True, False, "trilinear", "nearest-exact", False, None),
        # pytorch seg int16 with trilinear interpolation
        (ptint16seg, (75, 100, 100), True, False, "trilinear", "nearest-exact", False, None),
        (ptint16seg, (51, 131, 155), True, False, "trilinear", "nearest-exact", False, None),
        # pytorch seg int8 with trilinear interpolation with memefficient_seg_resampling
        (ptint8seg, (100, 128, 256), True, True, "trilinear", "nearest-exact", False, None),
        (ptint8seg, (51, 131, 155), True, True, "trilinear", "nearest-exact", False, None),
        # pytorch seg int16 with trilinear interpolation with memefficient_seg_resampling
        (ptint16seg, (75, 100, 100), True, True, "trilinear", "nearest-exact", False, None),
        (ptint16seg, (51, 131, 155), True, True, "trilinear", "nearest-exact", False, None),
        # pytorch seg int8 with trilinear interpolation
        (ptint8seg, (100, 128, 256), True, False, "nearest", "nearest-exact", False, None),
        (ptint8seg, (51, 131, 155), True, False, "nearest", "nearest-exact", False, None),
        # pytorch seg int16 with trilinear interpolation
        (ptint16seg, (75, 100, 100), True, False, "nearest", "nearest-exact", False, None),
        (ptint16seg, (51, 131, 155), True, False, "nearest", "nearest-exact", False, None),
        # pytorch seg int8 with trilinear interpolation with memefficient_seg_resampling
        (ptint8seg, (100, 128, 256), True, True, "nearest", "nearest-exact", False, None),
        (ptint8seg, (51, 131, 155), True, True, "nearest", "nearest-exact", False, None),
        # pytorch seg int16 with trilinear interpolation with memefficient_seg_resampling
        (ptint16seg, (75, 100, 100), True, True, "nearest", "nearest-exact", False, None),
        (ptint16seg, (51, 131, 155), True, True, "nearest", "nearest-exact", False, None),
        # make sure the it works for do_separate_z
        (npfloat32data, (100, 196, 196), False, False, "bilinear", "nearest-exact", True, [0]),
        (npfloat32data, (51, 131, 155), False, False, "bilinear", "nearest-exact", True, [1]),
        (npfloat16data, (100, 196, 128), False, False, "bilinear", "nearest-exact", True, [2]),
        (npfloat16data, (51, 100, 155), False, False, "bilinear", "nearest-exact", True, [0]),
        (npint8seg, (100, 128, 256), True, False, "bilinear", "nearest-exact", True, [1]),
        (npint8seg, (51, 131, 155), True, False, "bilinear", "nearest-exact", True, [2]),
        (npint16seg, (75, 100, 100), True, False, "bilinear", "nearest-exact", True, [0]),
        (npint16seg, (51, 131, 155), True, False, "bilinear", "nearest-exact", True, [1]),
        (npint8seg, (100, 128, 256), True, True, "bilinear", "nearest-exact", True, [2]),
        (npint8seg, (51, 131, 155), True, True, "bilinear", "nearest-exact", True, [0]),
        (npint16seg, (75, 100, 100), True, True, "bilinear", "nearest-exact", True, [1]),
        (npint16seg, (51, 131, 155), True, True, "bilinear", "nearest-exact", True, [2]),
        (npint8seg, (100, 128, 256), True, False, "nearest", "nearest-exact", True, [0]),
        (npint8seg, (51, 131, 155), True, False, "nearest", "nearest-exact", True, [1]),
        (npint16seg, (75, 100, 100), True, False, "nearest", "nearest-exact", True, [2]),
        (npint16seg, (51, 131, 155), True, False, "nearest", "nearest-exact", True, [0]),
        (npint8seg, (100, 128, 256), True, True, "nearest", "nearest-exact", True, [1]),
        (npint8seg, (51, 131, 155), True, True, "nearest", "nearest-exact", True, [2]),
        (npint16seg, (75, 100, 100), True, True, "nearest", "nearest-exact", True, [0]),
        (npint16seg, (51, 131, 155), True, True, "nearest", "nearest-exact", True, [1]),
        (ptfloat32data, (100, 196, 196), False, False, "bilinear", "nearest-exact", True, [1]),
        (ptfloat32data, (51, 131, 155), False, False, "bilinear", "nearest-exact", True, [2]),
        (ptfloat16data, (100, 196, 128), False, False, "bilinear", "nearest-exact", True, [0]),
        (ptfloat16data, (51, 100, 155), False, False, "bilinear", "nearest-exact", True, [1]),
        (ptint8seg, (100, 128, 256), True, False, "bilinear", "nearest-exact", True, [2]),
        (ptint8seg, (51, 131, 155), True, False, "bilinear", "nearest-exact", True, [0]),
        (ptint16seg, (75, 100, 100), True, False, "bilinear", "nearest-exact", True, [1]),
        (ptint16seg, (51, 131, 155), True, False, "bilinear", "nearest-exact", True, [2]),
        (ptint8seg, (100, 128, 256), True, True, "bilinear", "nearest-exact", True, [0]),
        (ptint8seg, (51, 131, 155), True, True, "bilinear", "nearest-exact", True, [1]),
        (ptint16seg, (75, 100, 100), True, True, "bilinear", "nearest-exact", True, [2]),
        (ptint16seg, (51, 131, 155), True, True, "bilinear", "nearest-exact", True, [0]),
        (ptint8seg, (100, 128, 256), True, False, "nearest", "nearest-exact", True, [1]),
        (ptint8seg, (51, 131, 155), True, False, "nearest", "nearest-exact", True, [2]),
        (ptint16seg, (75, 100, 100), True, False, "nearest", "nearest-exact", True, [0]),
        (ptint16seg, (51, 131, 155), True, False, "nearest", "nearest-exact", True, [1]),
        (ptint8seg, (100, 128, 256), True, True, "nearest", "nearest-exact", True, [2]),
        (ptint8seg, (51, 131, 155), True, True, "nearest", "nearest-exact", True, [0]),
        (ptint16seg, (75, 100, 100), True, True, "nearest", "nearest-exact", True, [1]),
        (ptint16seg, (51, 131, 155), True, True, "nearest", "nearest-exact", True, [2]),
    ],
)
def test_resample_data_or_seg_torch(
    data, new_shape, is_seg, memefficient_seg_resampling, mode, aniso_axis_mode, do_separate_z, axis
):
    resampled_data = resample_data_or_seg_torch(
        data,
        new_shape,
        is_seg,
        memefficient_seg_resampling=memefficient_seg_resampling,
        mode=mode,
        aniso_axis_mode=aniso_axis_mode,
        do_separate_z=do_separate_z,
        axis=axis,
    )
    assert resampled_data.shape[1:] == new_shape
    assert data.shape[0] == resampled_data.shape[0]
    assert data.dtype == resampled_data.dtype
