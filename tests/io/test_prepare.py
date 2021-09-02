import numpy as np
import SimpleITK as sitk

from nndet.io.prepare import split_4d_itk


def test_split_4d_itk():
    data = np.zeros((2, 10, 10, 10))

    slices = []
    for t in range(2):
        slices.append(sitk.GetImageFromArray(data[t], False))

    data_itk = sitk.JoinSeries(slices)

    origin = (100, 100, 100, 1)
    data_itk.SetOrigin(origin)
    spacing = (2.0, 3.0, 4.0, 5.0)
    data_itk.SetSpacing(spacing)
    direction = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.0,
        0.0,
        0.0,
        0.0,
        0.0,
        4.0,
    ]
    data_itk.SetDirection(direction)
    splitted_data_itk = split_4d_itk(data_itk)

    for sdi in splitted_data_itk:
        assert sdi.GetSpacing() == spacing[:-1]
        assert sdi.GetOrigin() == origin[:-1]
        assert sdi.GetDirection() == (1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0)
        assert sdi.GetDimension() == 3
    assert len(splitted_data_itk) == 2
