import numpy as np

from nndet.inference import restore_detection


def test_center_crop_object_mask():
    transpose_backward = [2, 0, 1]
    original_spacing = [2.0, 1.0, 1.0]
    spacing_after_resampling = [2.0, 2.0, 2.0]
    crop_bbox = [(2, 4), (10, 100), (10, 100)]

    boxes = np.array([[10.0, 10.0, 20.0, 20.0, 4.0, 6.0]])
    boxes_expected = np.array([[6.0, 30.0, 8.0, 50.0, 30.0, 50.0]])
    boxe_corrected = restore_detection(
        boxes,
        transpose_backward=transpose_backward,
        original_spacing=original_spacing,
        spacing_after_resampling=spacing_after_resampling,
        crop_bbox=crop_bbox,
    )
    assert np.allclose(boxes_expected, boxe_corrected)
