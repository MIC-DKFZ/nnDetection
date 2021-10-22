import numpy as np
import pytest
import torch
from batchgenerators.transforms.utility_transforms import (
    ConvertSegToBoundingBoxCoordinates,
)

from nndet.io.transforms.instances import (
    get_instance_class_from_properties,
    get_instance_class_from_properties_seq,
    instances_to_boxes,
    instances_to_boxes_np,
    instances_to_fg,
    instances_to_fg_np,
    instances_to_segmentation,
    instances_to_segmentation_np,
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
def box_result():
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


def seg_result():  # 1->1, 2->2, 3->1, 4->2
    seg = np.zeros((10, 10, 10))
    seg[0, 0, 0] = 1
    seg[2:4, 2:4, 1:3] = 2
    seg[5:7, 2:4, 3:8] = 1
    seg[8:10, 7:9, 2:6] = 2
    seg = seg[None]  # add channel
    return seg


def seg_bg_result():  # 1->1, 2->2, 3->1, 4->2
    seg = np.zeros((10, 10, 10))
    seg[0, 0, 0] = 2
    seg[2:4, 2:4, 1:3] = 3
    seg[5:7, 2:4, 3:8] = 2
    seg[8:10, 7:9, 2:6] = 3
    seg = seg[None]  # add channel
    return seg


@pytest.fixture
def fg_result():
    seg = np.zeros((10, 10, 10))
    seg[0, 0, 0] = 1
    seg[2:4, 2:4, 1:3] = 1
    seg[5:7, 2:4, 3:8] = 1
    seg[8:10, 7:9, 2:6] = 1
    seg = seg[None]  # add channel
    return seg


EXAMPLES = [
    (seg_result(), False),
    (seg_bg_result(), True),
]
SEG_MAPPINGS = [
    {"1": "1", "2": "2", "3": "1", "4": "2"},
    {1: 1, 2: 2, 3: 1, 4: 2},
]
INSTANCE_IDX = [
    None,
    [1, 2, 3, 4],
]
INSTANCE_CLASSES = [
    ([1], [1]),
    ([1, 2], [1, 2]),
    ([3], [1]),
    ([1, 2, 3, 4], [1, 2, 1, 2]),
    ([3, 2, 1, 1], [1, 2, 1, 1]),
]


def test_instance_to_boxes_np(mask, box_result):
    exptected_boxes, expected_instances = box_result
    boxes, inst = instances_to_boxes_np(mask, dim=3)
    assert np.allclose(boxes, exptected_boxes)
    assert np.allclose(inst, expected_instances)


def test_instance_to_boxes(mask, box_result):
    exptected_boxes, expected_instances = box_result

    mask = torch.from_numpy(mask).long()
    exptected_boxes = torch.from_numpy(exptected_boxes).float()
    expected_instances = torch.from_numpy(expected_instances)

    boxes, inst = instances_to_boxes(mask, dim=3)
    assert torch.allclose(boxes, exptected_boxes)
    assert torch.allclose(inst, expected_instances)


def test_instance_to_boxes_np_bg(mask, box_result):
    exptected_boxes, _ = box_result
    trafo = ConvertSegToBoundingBoxCoordinates(dim=3, get_rois_from_seg_flag=False)
    inp = {"seg": mask[None], "class_target": [[1, 1, 1, 1]]}
    res = trafo(**inp)
    assert np.allclose(res["bb_target"], exptected_boxes)


@pytest.mark.parametrize("seg_mapping", SEG_MAPPINGS)
@pytest.mark.parametrize("instance_idx", INSTANCE_IDX)
@pytest.mark.parametrize("seg_result,add_background", EXAMPLES)
def test_instances_to_segmentation(
    mask,
    seg_result,
    add_background,
    seg_mapping,
    instance_idx,
):
    mask = torch.from_numpy(mask)
    seg_result = torch.from_numpy(seg_result)
    if instance_idx is not None:
        instance_idx = torch.tensor(instance_idx)

    result = instances_to_segmentation(
        instances=mask,
        mapping=seg_mapping,
        instance_idx=None,
        add_background=add_background,
    )
    assert torch.allclose(result, seg_result)


@pytest.mark.parametrize("seg_mapping", SEG_MAPPINGS)
@pytest.mark.parametrize("seg_result,add_background", EXAMPLES)
def test_instances_to_segmentation_np(mask, seg_result, add_background, seg_mapping):
    result = instances_to_segmentation_np(
        instances=mask,
        mapping=seg_mapping,
        add_background=add_background,
    )
    assert np.allclose(result, seg_result)


def test_instances_to_fg(mask, fg_result):
    mask = torch.from_numpy(mask)
    fg_result = torch.from_numpy(fg_result)

    result = instances_to_fg(
        instances=mask,
    )
    assert torch.allclose(result, fg_result)


def test_instances_to_fg_np(mask, fg_result):
    result = instances_to_fg_np(
        instances=mask,
    )
    assert np.allclose(result, fg_result)


@pytest.mark.parametrize("map_dict", SEG_MAPPINGS)
@pytest.mark.parametrize("example", INSTANCE_CLASSES)
def test_get_instance_class_from_properties(example, map_dict):
    instances_idx, expected_classes = example

    instance_idx = torch.tensor(instances_idx)
    result = get_instance_class_from_properties(
        instance_idx=instance_idx,
        map_dict=map_dict,
    )
    assert torch.allclose(result, torch.tensor(expected_classes))


@pytest.mark.parametrize("map_dict", SEG_MAPPINGS)
@pytest.mark.parametrize("example", INSTANCE_CLASSES)
def test_get_instance_class_from_properties_seq(example, map_dict):
    instance_idx, expected_classes = example
    result = get_instance_class_from_properties_seq(
        instance_idx=instance_idx,
        map_dict=map_dict,
    )
    assert result == expected_classes
