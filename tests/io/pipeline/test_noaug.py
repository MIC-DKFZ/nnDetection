import pytest

from nndet.io.augmentation.pipeline.noaug import NoAug

PATCH_SIZES = [
    [128, 128, 128],
    [128, 128, 64],
    [128, 64, 128],
    [64, 128, 128],
    [64, 62, 60],
]

EXPECTED_ANY_AXES = [True, True, True, True, False]
EXPECTED_SAME_AXES = [[0, 1, 2], [0, 1], [0, 2], [1, 2], []]
params = {
    "do_dummy_2D_data_aug": False,
    "selected_seg_channels": None,
    "rotation_x": [30],
    "rotation_y": [30],
    "rotation_z": [30],
}


@pytest.mark.parametrize(
    "patch_size,expected_result",
    list(zip(PATCH_SIZES, EXPECTED_ANY_AXES)),
)
def test_any_matching_axes(patch_size, expected_result):
    pipeline = NoAug(patch_size, params)
    assert pipeline.any_matching_axes() == expected_result


@pytest.mark.parametrize(
    "patch_size,expected_result",
    list(zip(PATCH_SIZES[:-1], EXPECTED_SAME_AXES[:-1])),
)
def test_same_axes(patch_size, expected_result):
    pipeline = NoAug(patch_size, params)
    assert pipeline.same_axes() == expected_result
