import pytest
import torch

from nndet.inference.ensembler import OverlapMap


@pytest.fixture
def overlap_map():
    crop0 = [slice(10, 2), slice(0, 10), slice(-3, 8)]
    crop1 = [slice(0, 10), slice(3, 13)]

    overlap_map = OverlapMap((10, 10))
    overlap_map.add_overlap(crop0)
    overlap_map.add_overlap(crop1)
    return overlap_map


class TestOverlapMap:
    def test_add_overlap(self, overlap_map):
        expected_map = torch.ones(10, 10)
        expected_map[:, 3:8] += 1
        assert(overlap_map.overlap_map.allclose(expected_map))

    def test_mean_num_overlap_of_box_0(self, overlap_map):
        box0 = [2, 3, 7, 8]
        res = overlap_map.mean_num_overlap_of_box(box0)
        assert res == 2

    def test_mean_num_overlap_of_box_1(self, overlap_map):
        box0 = [2, 2, 8, 4]
        res = overlap_map.mean_num_overlap_of_box(box0)
        assert res == 1.5

    def test_mean_num_overlap_of_boxes(self, overlap_map):
        boxes = torch.tensor([[2, 3, 7, 8], [2, 2, 8, 4]])
        res = overlap_map.mean_num_overlap_of_boxes(boxes)
        expected = torch.tensor([2., 1.5])
        assert (res.allclose(expected))
