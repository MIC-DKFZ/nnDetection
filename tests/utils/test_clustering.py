import numpy as np
import pytest

from nndet.utils.clustering import (
    compute_score_from_seg,
    seg_to_instances,
    seg_to_instances_voted,
    softmax_to_instances,
)


def clusters():
    cl1 = (slice(2, 5), slice(2, 5))  # first cluster
    cl2 = (slice(4, 7), slice(4, 7))  # second cluster overlapped with first
    cl3 = (slice(1, 4), slice(8, 10))  # third cluster
    return cl1, cl2, cl3


@pytest.fixture
def seg():
    cl1, cl2, cl3 = clusters()
    seg = np.zeros((10, 10))
    seg[cl1] = 1
    seg[cl2] = 2
    seg[cl3] = 1
    return seg


@pytest.fixture
def expected_mask_no_cls():  # expected mask without class info
    cl1, cl2, cl3 = clusters()
    expected_mask_no_cls = np.zeros((10, 10))
    expected_mask_no_cls[cl1] = 2
    expected_mask_no_cls[cl2] = 2
    expected_mask_no_cls[cl3] = 1
    return expected_mask_no_cls


@pytest.fixture
def expected_mask_cls():  # expected mask without class info
    cl1, cl2, cl3 = clusters()
    expected_mask_cls = np.zeros((10, 10))
    expected_mask_cls[cl1] = 2
    expected_mask_cls[cl2] = 3
    expected_mask_cls[cl3] = 1
    return expected_mask_cls


@pytest.fixture
def expected_boxes_no_cls():
    boxes = np.array(
        [
            [0, 7, 4, 10],
            [1, 1, 7, 7],
        ]
    )
    labels = np.array([0, 1])
    return boxes, labels


def test_seg_to_instances(seg, expected_mask_cls):
    mask, mask_cls = seg_to_instances(seg)
    assert np.allclose(mask, expected_mask_cls)
    assert mask_cls == {1: 1, 2: 1, 3: 2}


def test_seg_to_instances_min_vox(seg):
    cl1, cl2, _ = clusters()
    expected_mask = np.zeros((10, 10))
    expected_mask[cl1] = 1
    expected_mask[cl2] = 2

    mask, mask_cls = seg_to_instances(seg, min_num_voxel=7)
    assert np.allclose(mask, expected_mask)
    assert mask_cls == {1: 1, 2: 2}


def test_seg_to_instances_voted(seg, expected_mask_no_cls):
    mask, mask_cls = seg_to_instances_voted(seg)
    assert np.allclose(mask, expected_mask_no_cls)
    assert mask_cls == {1: 1, 2: 2}


def test_seg_to_instances_voted_min_vox(seg):
    cl1, cl2, _ = clusters()
    expected_mask = np.zeros((10, 10))
    expected_mask[cl1] = 1
    expected_mask[cl2] = 1

    mask, mask_cls = seg_to_instances_voted(seg, min_num_voxel=7)
    assert np.allclose(mask, expected_mask)
    assert mask_cls == {1: 2}


def test_compute_score_from_seg():
    cl1, cl2, cl3 = clusters()
    mask = np.zeros((10, 10))
    mask[cl1] = 3
    mask[cl2] = 2
    mask[cl3] = 1

    mask_classes = {1: 1, 2: 2, 3: 1}

    probs = np.random.rand(3, 10, 10) * 0.1

    probs[(1, *cl3)] = 1.0
    probs[(2, *cl2)] = 2.0
    probs[(1, *cl1)] = 3.0

    scores = compute_score_from_seg(mask, mask_classes, probs, "mean")
    assert np.allclose(scores, [1.0, 2.0, 3.0])


def test_softmax_to_instances(expected_boxes_no_cls):
    cl1, cl2, cl3 = clusters()
    expected_boxes, expected_labels = expected_boxes_no_cls

    probs = np.zeros((3, 10, 10)) * 0.1

    probs[(1, *cl1)] = 0.8
    probs[(2, *cl2)] = 0.9
    probs[(1, *cl3)] = 0.4

    res = softmax_to_instances(probs, aggregation="mean", min_threshold=0.33)
    exp_prob = (9 * 0.9) / (8 + 9)

    assert np.allclose(res["pred_boxes"], expected_boxes)
    assert np.allclose(res["pred_labels"], expected_labels)
    assert np.allclose(res["pred_scores"], [0.4, exp_prob])
