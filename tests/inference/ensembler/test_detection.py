import pytest
import copy
from pathlib import Path

import torch
import os
from tempfile import TemporaryDirectory

from nndet.inference.ensembler import BoxEnsembler
from unittest.mock import Mock, patch
from dataclasses import dataclass


@dataclass
class Example:
    case = {"data": torch.zeros(1, 10, 10)}
    properties = {
            "transpose_backward": (0, 1),
            "original_spacing": (1.0, 1.0),
            "spacing_after_resampling": (1., 1.),
            "crop_bbox": (0, 10, 0, 10,),
            "original_size_of_raw_data": [100, 100],
            "itk_origin": 0,
            "itk_spacing": 1,
            "itk_direction": -1,
        }
    batch0 = {
        "data": torch.zeros(2, 1, 5, 5),
        "tile_origin": [[0], [0]],
        "crop": [[...], [slice(0, 6)], [slice(0, 6)]],
        }
    result0 = {
        "pred_boxes": [torch.tensor([[0, 0, 1, 1]]).float()],
        "pred_scores": [torch.tensor([1.])],
        "pred_labels": [torch.tensor([1])],
        }
    batch1 = copy.deepcopy(batch0)
    batch1["tile_origin"] = [[5], [5]]
    result1 = copy.deepcopy(result0)


@pytest.fixture
def example():
    return Example()


class TestDetectionEnsembler:
    def test_from_case(self, example):
        ensembler = BoxEnsembler.from_case(
            case=example.case,
            properties=example.properties,
            parameters={"model_iou": 0.5, "ensemble_topk": 10}
        )
        expected_shape = list(example.case["data"].shape)[1:]

        assert all([a == b for a, b in zip(ensembler.properties["shape"], expected_shape)])
        assert all([a == b for a, b in zip(ensembler.properties["transpose_backward"], (0, 1))])
        assert ensembler.parameters["model_iou"] == 0.5
        assert ensembler.parameters["ensemble_topk"] == 10

    def test_add_model(self, example):
        ensembler = BoxEnsembler.from_case(
            case=example.case,
            properties=example.properties,
            parameters={},
        )
        ensembler.add_model("model_test", 0.5)
        
        assert ensembler.model_current == "model_test"
        assert ensembler.model_weights["model_test"] == 0.5
        assert "model_test" in ensembler.model_results
        assert "model_test" in ensembler.model_weights

    def test_process_batch(self, example):
        ensembler = BoxEnsembler.from_case(
            case=example.case,
            properties=example.properties,
            parameters={},
        )
        ensembler.add_model("model_test0", 1.)
        ensembler.process_batch(example.result0, example.batch0)
        ensembler.process_batch(example.result1, example.batch1)
        ensembler.add_model("model_test1", 1.)
        ensembler.process_batch(example.result0, example.batch0)

        expected_boxes0 = [torch.tensor([[0, 0, 1, 1]]).float(),
                          torch.tensor([[5, 5, 6, 6]]).float(),
                          ]
        expected_boxes1 = [torch.tensor([[0, 0, 1, 1]]).float(),]

        for exp_box, ens_box in zip(expected_boxes0, ensembler.model_results["model_test0"]["boxes"]):
            assert (exp_box.allclose(ens_box))
        for exp_box, ens_box in zip(expected_boxes1, ensembler.model_results["model_test1"]["boxes"]):
            assert (exp_box.allclose(ens_box))

    def test_get_box_in_tile_weight(self, example):
        ensembler = BoxEnsembler.from_case(
            case=example.case,
            properties=example.properties,
            parameters={},
        )

        tile_size = (10, 10)
        box_centers = torch.tensor([[5., 5.], [5., 5.]])
        pred_weight = ensembler._get_box_in_tile_weight(box_centers, tile_size)
        expected_weight = torch.tensor([1., 1.])
        assert pred_weight.allclose(expected_weight)

    def test_apply_offsets_to_boxes(self, example):
        ensembler = BoxEnsembler.from_case(
            case=example.case,
            properties=example.properties,
            parameters={},
        )

        boxes = [torch.tensor([[0, 0, 1, 1, 0, 1]]).float(),
                 torch.tensor([[0, 0, 1, 1, 0, 1]]).float()]
        offsets = [[0, 0, 0], [1, 2, 3]]
        res = ensembler._apply_offsets_to_boxes(boxes, offsets)
        expected = [torch.tensor([[0, 0, 1, 1, 0, 1]]).float(),
                    torch.tensor([[1, 2, 2, 3, 3, 4]]).float()]
        for r, e in zip(res, expected):
            assert r.allclose(e)

    def test_save_case_result(self, example):
        ensembler = BoxEnsembler.from_case(
            case=example.case,
            properties=example.properties,
            parameters={},
        )
        ensembler1 = BoxEnsembler.from_case(
            case=example.case,
            properties=example.properties,
            parameters={},
        )
        with TemporaryDirectory(dir=os.getcwd()) as _dir:
            ensembler.save_state(_dir, "tmp_case")
            ensembler1.load_state(Path(_dir), "tmp_case")

    def test_get_case_result(self, example):
        ensembler = BoxEnsembler.from_case(
            case=example.case,
            properties=example.properties,
            parameters={},
        )
        ensembler.add_model("model0", 1.0)
        ensembler.process_batch(example.result0, example.batch0)
        ensembler.add_model("model1", 0.5)
        ensembler.process_batch(example.result1, example.batch1)
        res = ensembler.get_case_result()

        expected_boxes = torch.tensor([[0, 0, 1, 1], [5, 5, 6, 6]]).float()
        expected_scores = torch.tensor([0.5, 0.5])
        expected_labels = torch.tensor([1., 1.])
        assert (res["pred_boxes"].allclose(expected_boxes))
        assert (res["pred_scores"].allclose(expected_scores))
        assert (res["pred_labels"].allclose(expected_labels))
