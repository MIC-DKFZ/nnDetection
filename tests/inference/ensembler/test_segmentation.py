import copy
import os
from dataclasses import dataclass
from tempfile import TemporaryDirectory

import pytest
import torch

from nndet.inference.ensembler import SegmentationEnsembler


@dataclass
class Example:
    case = {"data": torch.zeros(1, 10, 10)}
    properties = {
        "transpose_backward": (0, 1),
        "original_spacing": (1.0, 1.0),
        "spacing_after_resampling": (1.0, 1.0),
        "crop_bbox": (
            0,
            10,
            0,
            10,
        ),
        "size_after_cropping": [100, 100],
        "original_size_of_raw_data": [100, 100],
        "itk_origin": 0,
        "itk_spacing": 1,
        "itk_direction": -1,
    }
    batch0 = {"crop": [[...], [slice(0, 10)], [slice(0, 8)]]}
    result0 = {"pred_seg": torch.ones(1, 3, 10, 8)}
    batch1 = copy.deepcopy(batch0)
    batch1["crop"] = [[...], [slice(0, 10)], [slice(2, 10)]]
    result1 = {"pred_seg": torch.ones(1, 3, 10, 8)}


@pytest.fixture
def example():
    return Example()


class TestSegmentationEnsembler:
    def test_from_case(self, example):
        ensembler = SegmentationEnsembler.from_case(
            case=example.case,
            properties=example.properties,
            parameters={"model_iou": 0.5, "ensemble_topk": 10},
        )
        expected_shape = list(example.case["data"].shape)[1:]

        assert all(
            [a == b for a, b in zip(ensembler.properties["shape"], expected_shape)]
        )
        assert all(
            [a == b for a, b in zip(ensembler.properties["transpose_backward"], (0, 1))]
        )
        assert (ensembler.parameters["model_iou"], 0.5)
        assert (ensembler.parameters["ensemble_topk"], 10)

    def test_process_batch(self, example):
        ensembler = SegmentationEnsembler.from_case(
            case=example.case,
            properties=example.properties,
            parameters={},
        )
        ensembler.add_model(model_weight=1.0)
        ensembler.process_batch(example.result0, example.batch0)
        ensembler.process_batch(example.result1, example.batch1)

        w1 = ensembler.get_weighting((10, 8))
        w2 = ensembler.get_weighting((10, 8))
        expected_overlap = torch.zeros(10, 10)
        expected_overlap[:, :8] += w1
        expected_overlap[:, 2:] += w2

        assert ensembler.overlap.allclose(expected_overlap)

        result = ensembler.model_results / ensembler.overlap[None]
        assert result.allclose(torch.ones_like(result))

    def test_save_case_result(self, example):
        ensembler = SegmentationEnsembler.from_case(
            case=example.case,
            properties=example.properties,
            parameters={},
        )
        with TemporaryDirectory(dir=os.getcwd()) as _dir:
            ensembler.save_state(_dir, "tmp_case")

    def test_crop_to_case_boundaries(self, example):
        ensembler = SegmentationEnsembler.from_case(
            case={"data": torch.zeros(3, 10, 10)},
            properties=example.properties,
            parameters={},
        )
        ensembler.model_results = torch.zeros(3, 10, 10)

        crop = [slice(-3, 4), slice(7, 14)]
        seg = torch.rand(3, 7, 7)
        new_seg, slicer = ensembler.crop_to_case_boundaries(seg, crop)
        print(new_seg.shape)

        assert new_seg.allclose(seg[:, 3:, :3])
        assert all([a == b for a, b in zip((slicer[1].start, slicer[1].stop), (0, 4))])
        assert all([a == b for a, b in zip((slicer[2].start, slicer[2].stop), (7, 10))])
        ensembler.model_results[slicer] = new_seg

    def test_get_weighting(self, example):
        ensembler = SegmentationEnsembler.from_case(
            case={"data": torch.zeros(3, 10, 10)},
            properties=example.properties,
            parameters={"use_gaussian": True},
        )
        assert ensembler.parameters["use_gaussian"]
        w = ensembler.get_weighting((10, 10))
        assert tuple(w.shape) == (10, 10)
        w2 = ensembler.get_weighting((10, 10))

    def test_get_case_result(self, example):
        ensembler = SegmentationEnsembler.from_case(
            case={"data": torch.zeros(3, 10, 10)},
            properties=example.properties,
            parameters={"argmax": False},
        )
        assert not ensembler.parameters["argmax"]

        ensembler.model_results = torch.ones(3, 10, 10)
        ensembler.overlap = torch.ones(10, 10)
        res = ensembler.get_case_result()["pred_seg"]
        assert res.allclose(torch.ones(3, 10, 10))

        ensembler.model_results[1:] *= 0.5
        ensembler.update_parameters(argmax=True)
        res = ensembler.get_case_result()["pred_seg"]
        assert res.allclose(torch.zeros(10, 10, dtype=torch.uint8))
