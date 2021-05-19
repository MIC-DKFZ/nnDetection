import pytest
import numpy as np
import torch

from nndet.io.patching import center_crop_object_mask, \
    center_crop_object_seg, save_get_crop, create_grid
from nndet.io.transforms.instances import instances_to_boxes


@pytest.fixture
def mask():
    mask = np.zeros((100, 100))
    mask[10:20, 10:20] = 1
    mask[60:80, 40:60] = 2
    return mask


@pytest.fixture
def seg():
    seg = np.zeros((100, 100))
    seg[10:20, 10:20] = 1
    seg[60:80, 40:60] = 1
    return seg


def _check_center_crops(crops):
    assert (slice(-6, 34), slice(-6, 34)) in crops
    assert (slice(49, 89), slice(29, 69)) in crops
    assert len(crops) == 2


def _check_save_get(data, crop, data_gt, **kwargs):
    _data = save_get_crop(data, crop, **kwargs)[0]
    assert (_data == data_gt).all()


def test_center_crop_object_mask(mask):
    # test tuple shape
    crops = center_crop_object_mask(mask, (40, 40))
    _check_center_crops(crops)

    # test int shape
    crops = center_crop_object_mask(mask, 40)
    _check_center_crops(crops)

    # check empty mask
    assert not center_crop_object_mask(np.zeros((100, 100)), 40)


def test_center_crop_object_mask_errors(mask):
    with pytest.raises(TypeError):
        center_crop_object_mask(mask, (10,))

    with pytest.raises(TypeError):
        center_crop_object_mask(mask, 200)


def test_center_crop_object_seg(seg):
    # test tuple shape
    crops = center_crop_object_seg(seg, (40, 40))
    _check_center_crops(crops)

    # test int shape
    crops = center_crop_object_seg(seg, 40)
    _check_center_crops(crops)

    # check empty mask
    assert not center_crop_object_seg(np.zeros((100, 100)), 40)


def test_center_crop_object_seg_errors(seg):
    with pytest.raises(TypeError):
        center_crop_object_seg(seg, (10,))

    with pytest.raises(TypeError):
        center_crop_object_seg(seg, 200)


def test_create_grid_fixed():
    # test fixed mode
    crops = create_grid(40, (60, 60), 20, mode='fixed')
    assert ((slice(0, 40), slice(0, 40)) in crops)
    assert ((slice(20, 60), slice(0, 40)) in crops)
    assert ((slice(0, 40), slice(20, 60)) in crops)
    assert ((slice(20, 60), slice(20, 60)) in crops)
    assert len(crops) == 4


def test_create_grid_fixed_psize_dsize():
    crops = create_grid(40, (40, 40), 20, mode='fixed')
    assert ((slice(0, 40), slice(0, 40)) in crops)
    assert len(crops) == 1


def test_create_grid_symmetric():
    crops = create_grid((40, 40), (50, 50), (20, 20), mode='symmetric')
    assert ((slice(-15, 25), slice(-15, 25)) in crops)
    assert ((slice(-15, 25), slice(5, 45)) in crops)
    assert ((slice(-15, 25), slice(25, 65)) in crops)

    assert ((slice(5, 45), slice(-15, 25)) in crops)
    assert ((slice(5, 45), slice(5, 45)) in crops)
    assert ((slice(5, 45), slice(25, 65)) in crops)

    assert ((slice(25, 65), slice(-15, 25)) in crops)
    assert ((slice(25, 65), slice(5, 45)) in crops)
    assert ((slice(25, 65), slice(25, 65)) in crops)

    assert len(crops) == 9


def test_create_grid_synmmetric_psize_dsize():
    crops = create_grid((60, 40), (40, 40), 20, mode='symmetric')
    assert ((slice(-10, 50), slice(0, 40)) in crops)
    assert len(crops) == 1


def test_create_grid_errors():
    with pytest.raises(TypeError):
        create_grid((40,), (50, 50), (20, 20))

    with pytest.raises(TypeError):
        create_grid((40, 40), (50, 50), (20,))

    with pytest.raises(TypeError):
        create_grid((40, 40), (50, 50), (50, 20))


def test_save_get_borders():
    mask = np.zeros((20, 20))
    top_left = (slice(-10, 10), slice(-10, 10))
    top_left_gt = np.zeros((20, 20)) + 2
    top_left_gt[10:, 10:] = 0
    _check_save_get(mask, top_left, top_left_gt, mode='constant', constant_values=2)

    top_right = (slice(-10, 10), slice(10, 30))
    top_right_gt = np.zeros((20, 20)) + 2
    top_right_gt[10:, :10] = 0
    _check_save_get(mask, top_right, top_right_gt, mode='constant', constant_values=2)

    bottom_left = (slice(10, 30), slice(-10, 10))
    bottom_left_gt = np.zeros((20, 20)) + 2
    bottom_left_gt[:10, 10:] = 0
    _check_save_get(mask, bottom_left, bottom_left_gt, mode='constant', constant_values=2)

    bottom_right = (slice(10, 30), slice(10, 30))
    bottom_right_gt = np.zeros((20, 20)) + 2
    bottom_right_gt[:10, :10] = 0
    _check_save_get(mask, bottom_right, bottom_right_gt, mode='constant', constant_values=2)


def test_save_get_mode_shift():
    mask = np.zeros((20, 20))
    mask[15:, 15:] = 1
    gt = np.zeros((20, 20))
    gt[15:, 15:] = 1
    crop = (slice(10, 30), slice(-10, 10))
    _check_save_get(mask, crop, gt, mode='shift')


def test_save_get_extra_dim_compension():
    mask = np.zeros((1, 20, 20))
    mask[0, 15:, 15:] = 1
    gt = np.zeros((1, 20, 20))
    gt[0, 15:, 15:] = 1
    crop = (slice(10, 30), slice(10, 30))
    _check_save_get(mask, crop, gt, mode='shift')


def test_save_get_errors():
    with pytest.raises(RuntimeError):
        save_get_crop(np.zeros((10, 10)),
                      (slice(0, 20), slice(0, 5)), mode='shift')

    with pytest.raises(RuntimeError):
        save_get_crop(np.zeros((10, 10)),
                      (slice(0, 5), slice(-10, 5)), mode='shift')

    with pytest.raises(TypeError):
        save_get_crop(np.zeros((10, 10)),
                      (slice(0, 5), slice(-10, 5), slice(-10, 5)),
                      mode='shift')


def test_save_get_origin_shifted():
    data = np.random.random((3, 10, 10, 10))
    crop, origin, _ = save_get_crop(data, crop=(slice(-2, 1), slice(-3, 1), slice(1, 4)))
    assert all([a == b for a, b in zip(origin, [0, 0, 1])])


def test_save_get_origin_pad():
    data = np.random.random((3, 10, 10, 10))
    crop, origin, _ = save_get_crop(data,
                                    crop=(slice(-2, 1), slice(-3, 1), slice(1, 4)),
                                    mode='constant',
                                    )
    assert all([a == b for a, b in zip(origin, [-2, -3, 1])])


def test_integration_origin_bounding_box_offset():
    data = np.zeros((1, 3, 10, 10, 10))
    data[..., 1:4, 1:4, 1:4] = 1
    crop, origin, _ = save_get_crop(data,
                                    crop=(slice(-2, 5), slice(-2, 5), slice(-2, 5)),
                                    mode='constant',
                                    )
    expected_bbox = instances_to_boxes(torch.from_numpy(data), 3)[0]
    predicted_box = instances_to_boxes(torch.from_numpy(crop), 3)[0]
    offset = torch.Tensor(origin)

    predicted_box[:, 0] += offset[0]
    predicted_box[:, 1] += offset[1]
    predicted_box[:, 2] += offset[0]
    predicted_box[:, 3] += offset[1]
    predicted_box[:, 4] += offset[2]
    predicted_box[:, 5] += offset[2]

    assert expected_bbox.allclose(predicted_box)
