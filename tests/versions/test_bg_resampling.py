import numpy as np

from nndet.preprocessing.resampling import resample_patient


def test_resample_patient_smoke():
    np.random.seed(0)
    data = np.random.rand(2, 64, 64, 64)
    seg = np.random.rand(2, 64, 64, 64)
    original_spacing = (1.0, 1.0, 1.0)
    target_spacing = (0.5, 2.0, 2.0)

    data_res, seg_res = resample_patient(
        data=data,
        seg=seg,
        original_spacing=original_spacing,
        target_spacing=target_spacing,
        order_data=3,
        order_seg=0,
        force_separate_z=False,
        order_z_data=0,
        order_z_seg=0,
    )
