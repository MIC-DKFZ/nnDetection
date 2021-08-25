import pytest

import torch
import numpy as np
from nndet.io.transforms.instances import instances_to_boxes, instances_to_boxes_np
from batchgenerators.transforms.utility_transforms import (
    ConvertSegToBoundingBoxCoordinates,
)


@pytest.fixture
def mask():
    mask = np.zeros((10, 10, 10))
    mask[0, 0, 0] = 1
    mask[2:4, 2:4, 1:3] = 2
    mask[5:7, 2:4, 3:8] = 3
    mask[8:10, 7:9, 2:6] = 4
    mask = mask[None]  # add channel
    return mask


@pytest.fixture
def result():
    boxes = np.array(
        [
            [-1, -1, 1, 1, -1, 1],
            [1, 1, 4, 4, 0, 3],
            [4, 1, 7, 4, 2, 8],
            [7, 6, 10, 9, 1, 6],
        ]
    )
    inst = np.array([1, 2, 3, 4])
    return boxes, inst


def test_instance_to_boxes_np(mask, result):
    exptected_boxes, expected_instances = result
    boxes, inst = instances_to_boxes_np(mask, dim=3)
    assert np.allclose(boxes, exptected_boxes)
    assert np.allclose(inst, expected_instances)


def test_instance_to_boxes(mask, result):
    exptected_boxes, expected_instances = result

    mask = torch.from_numpy(mask).long()
    exptected_boxes = torch.from_numpy(exptected_boxes).float()
    expected_instances = torch.from_numpy(expected_instances)

    boxes, inst = instances_to_boxes(mask, dim=3)
    assert torch.allclose(boxes, exptected_boxes)
    assert torch.allclose(inst, expected_instances)


def test_instance_to_boxes_np_bg(mask, result):
    exptected_boxes, _ = result
    trafo = ConvertSegToBoundingBoxCoordinates(dim=3, get_rois_from_seg_flag=False)
    inp = {"seg": mask[None], "class_target": [[1, 1, 1, 1]]}
    res = trafo(**inp)
    assert np.allclose(res["bb_target"], exptected_boxes)
