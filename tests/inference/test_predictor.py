# from typing import Dict, Optional

# import torch
# import numpy as np
# import unittest

# from unittest.mock import Mock

# from nndet.inference.predictor import Predictor
# from nndet.inference.ensembler import BaseEnsembler

# import nndet.inference.registry
# from nndet import PREDICTOR_REGISTRY


# def get_dummy_model():
#     from nndet.io.transforms.instances import instances_to_boxes
#     class GetForeground(torch.nn.Module):
#         def __init__(self, *args, **kwargs):
#             super().__init__()

#         def forward(self, *input, **kwargs):
#             return super().forward(*input, **kwargs)

#         def inference_step(self, images, **kwargs):
#             boxes = [instances_to_boxes(img, dim=(images.ndimension() - 2))[0]
#                      for img in images.split(1)]
#             probs = [torch.tensor([1. for box in _boxes]) for _boxes in boxes]
#             labels = [torch.tensor([1. for box in _boxes]) for _boxes in boxes]

#             pred_seg = torch.zeros(images.shape[0], 2, *images.shape[2:])
#             pred_seg[:, 1] = images[:, 0] > 0

#             return {"pred_boxes": boxes, "pred_scores": probs,
#                     "pred_labels": labels, "pred_seg": pred_seg}
#     return GetForeground()


# def get_dummy_data_left_top():
#     data = np.zeros((1, 10, 10))
#     data[:, 2:4, 2:4] = 1
#     return data


# def get_dummy_data_center():
#     data = np.zeros((1, 10, 10))
#     data[:, 4:7, 4:7] = 1
#     return data


# class DummyEnsembler(BaseEnsembler):
#     def __init__(self, case=None, properties=None):
#         super().__init__({}, {})
#         self.model = Mock()
#         self.process = Mock()
#         self.save = Mock()
#         self.result = Mock()

#     def add_model(self,
#                   name: Optional[str] = None,
#                   model_weight: Optional[float] = None,
#                   ) -> str:
#         self.model(name, model_weight)

#     def sweep_parameters(cls):
#         return {}

#     def process_batch(self, result, batch, model_weight: float = None):
#         self.process(result, batch, model_weight)

#     def save_case_result(self, target_dir, name, **kwargs):
#         self.save(target_dir, name)

#     def get_case_result(self):
#         self.result()


# class DummyModel:
#     def __init__(self):
#         self.called = Mock()

#     def __call__(self, *args, **kwargs):
#         self.called(*args, **kwargs)

#     def inference_step(self, *args, **kwargs):
#         self.called(*args, **kwargs)
#         return {"pred_res": torch.tensor([0])}

#     def to(self, *args, **kwargs):
#         pass

#     def eval(self):
#         pass

#     def cpu(self):
#         pass


# class TestPredictor(unittest.TestCase):
#     def setUp(self) -> None:
#         self.ensemble = [DummyModel(), DummyModel()]
#         self.ensembler = {"dummy0": DummyEnsembler,
#                           "dummy1": DummyEnsembler}
#         self.patch_size = (10, 10)
#         device = "cuda:0" if torch.cuda.device_count() != 0 else "cpu"
#         self.predictor = Predictor(
#             self.ensembler, self.ensemble, self.patch_size, device=device)
#         self.case = {
#             "data": torch.arange(0, 400).reshape(20, 20)[None],
#             "props": [0, 1, 2]
#         }
#         self.properties = {
#             "transpose_backward": (0, 1),
#             "original_spacing": (1.0, 1.0),
#             "spacing_after_resampling": (1., 1.),
#             "crop_bbox": (0, 20, 0, 20,),
#         }

#     def test_integration_predict_case(self):
#         self.predictor.predict_case(self.case, properties=self.properties)

#     def test_tile_case(self):
#         tiles = self.predictor.tile_case(self.case)
#         self.assertEqual(tiles[0]["props"], self.case["props"])
#         for tile in tiles:
#             self.assertEqual(tuple(tile["data"].shape[-len(self.patch_size):]), self.patch_size)
#             self.assertIn("tile_origin", tile)
#             self.assertIn("crop", tile)

#     def test_predict_tiles(self):
#         def _tta_predict_mock(model, batch, batch_num, model_weight):
#             model(batch)
#         self.predictor.tta_predict = _tta_predict_mock
#         tiles = [{"data": [0, 1, 2], "crops": slice(0, 10)}] * 5
#         self.predictor.predict_tiles(tiles)
#         for model in self.ensemble:
#             model.called.assert_called()

#     def test_predict_with_transformation(self):
#         pre_transform = Mock(return_value={"data": 1})
#         post_transform = Mock(return_value={"data": 2})

#         t = DummyEnsembler()
#         self.predictor.ensembler["det"] = t

#         dummy_model = DummyModel()
#         self.predictor.predict_with_transformation(
#             model=dummy_model, batch={"data": 0}, batch_num=0,
#             transform=pre_transform, inverse_transform=post_transform,
#             )

#         t.process.assert_called()
#         pre_transform.assert_called_with(data=0)
#         dummy_model.called.assert_called_with(1, batch_num=0)
#         post_transform.assert_called_with(pred_res=torch.tensor([0]))

#     def test_integration_detection_top_left(self):
#         predictor_cls = PREDICTOR_REGISTRY.get("BoxPredictor")
#         plan = {"patch_size": (8, 8), "batch_size": 2}
#         predictor = predictor_cls(plan=plan,
#                                   models=[get_dummy_model()],
#                                   num_tta_transforms=1,
#                                   )
#         result = predictor.predict_case({"data": get_dummy_data_left_top()}, properties=self.properties)
#         pred_boxes = result["det"]["pred_boxes"]
#         pred_scores = result["det"]["pred_scores"]
#         pred_labels = result["det"]["pred_labels"]
#         assert (pred_boxes.allclose(torch.tensor([[1., 1., 4., 4.]]).to(pred_boxes)))
#         assert (pred_scores.allclose(torch.tensor([1.]).to(pred_scores)))
#         assert (pred_labels.allclose(torch.tensor([1.]).to(pred_labels)))

#     def test_integration_detection_center(self):
#         predictor_cls = PREDICTOR_REGISTRY.get("BoxPredictor")
#         plan = {"patch_size": (5, 5), "batch_size": 2}
#         predictor = predictor_cls(plan=plan, models=[get_dummy_model()],
#                                   dim=2, num_tta_transforms=1)
#         result = predictor.predict_case({"data": get_dummy_data_center()}, properties=self.properties)
#         pred_boxes = result["det"]["pred_boxes"]
#         pred_scores = result["det"]["pred_scores"]
#         pred_labels = result["det"]["pred_labels"]
#         # because of the patch based inference, the bounding box is not
#         # the ground truth
#         # self.assertTrue(pred_boxes.allclose(torch.tensor([[1., 1., 4., 4.]]).to(pred_boxes)))
#         assert (pred_scores.allclose(torch.tensor([1.]).to(pred_scores)))
#         assert (pred_labels.allclose(torch.tensor([1.]).to(pred_labels)))


from functools import partial
from typing import Tuple

import numpy as np
import pytest
import torch

from nndet.inference.ensembler.detection import BoxEnsemblerSelective
from nndet.inference.ensembler.segmentation import SegmentationEnsembler
from nndet.inference.predictor import Predictor
from nndet.io.transforms.instances import instances_to_boxes, instances_to_boxes_np


@pytest.fixture
def properties_simple():
    return {
        "transpose_backward": [0, 1, 2],
        "original_spacing": [1.0, 1.0, 1.0],
        "spacing_after_resampling": [1.0, 1.0, 1.0],
        "crop_bbox": None,
        "size_after_cropping": None,
        "original_size_of_raw_data": None,
        "itk_origin": None,
        "itk_spacing": None,
        "itk_direction": None,
    }


class DummyBoxModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, dims]
        # return: [N, classes, dims]
        # print(x)
        return x.max(dim=1, keepdim=True)[0] > 0

    def inference_step(self, images, *args, **kwargs):
        pboxes = []
        pscores = []
        plabels = []

        for i in range(images.shape[0]):
            boxes, _ = instances_to_boxes(images[i].to(torch.int), dim=3)

            if boxes.nelement() > 0:
                scores = torch.tensor([1.0] * boxes.shape[0])
                labels = torch.tensor([1] * boxes.shape[0])
            else:
                scores = torch.tensor([])
                labels = torch.tensor([])

            pboxes.append(boxes.view(-1, (images.ndim - 2) * 2))
            pscores.append(scores)
            plabels.append(labels)

        return {
            "pred_boxes": pboxes,
            "pred_scores": pscores,
            "pred_labels": plabels,
        }


class DummySegModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, dims]
        # return: [N, classes, dims]
        # print(x)
        return x.max(dim=1, keepdim=True)[0] > 0

    def inference_step(self, images, *args, **kwargs):
        pred = self(images) * 1.0
        return {"pred_seg": pred}


SMOKE_SHAPES = [
    ((1, 256, 256, 256), (128, 128, 128)),
    ((1, 32, 256, 256), (128, 128, 128)),
    ((1, 256, 32, 256), (128, 128, 128)),
    ((1, 256, 256, 32), (128, 128, 128)),
    ((1, 256, 256, 256), (64, 128, 128)),
    ((1, 256, 256, 256), (128, 64, 128)),
    ((1, 256, 256, 256), (128, 128, 64)),
    ((1, 32, 32, 32), (128, 128, 128)),
    ((1, 32, 32, 32), (32, 32, 32)),
]


class TestPredictorSegmentationEnsembler:
    @pytest.mark.parametrize("shape,crop_size", SMOKE_SHAPES)
    @pytest.mark.parametrize("use_gaussian", [True, False])
    def test_integration_segmentation(
        self,
        properties_simple: dict,
        shape: Tuple,
        crop_size: Tuple,
        use_gaussian: bool,
    ):
        data = np.zeros(shape)
        idx = (slice(0, 1), *[slice(0, s // 2) for s in shape[1:]])
        data[idx] = 1
        assert data.max() == 1

        case = {"data": data}
        _fn, _fn_key = SegmentationEnsembler.constructor(
            parameters={"use_gaussian": use_gaussian, "argmax": False},
        )

        predictor = Predictor(
            ensembler={_fn_key: _fn},
            models=[DummySegModel()],
            crop_size=crop_size,
            device="cpu",
        )
        prediction = predictor.predict_case(case=case, properties=properties_simple)

        assert "seg" in prediction
        assert "pred_seg" in prediction["seg"]
        assert not prediction["seg"]["restore"]
        assert np.allclose(data, prediction["seg"]["pred_seg"].numpy())


class TestPredictorBoxEnsembler:
    @pytest.mark.parametrize("shape,crop_size", SMOKE_SHAPES)
    @pytest.mark.parametrize("obj_scale", [2.0, 3.0])
    @pytest.mark.parametrize("obj_at_origin", [True, False])
    def test_integration_boxes(
        self,
        properties_simple: dict,
        shape: Tuple,
        crop_size: Tuple,
        obj_scale: float,
        obj_at_origin: bool,
    ):
        data = np.zeros(shape)
        if obj_at_origin:
            idx = (slice(0, 1), *[slice(0, int(s / obj_scale)) for s in crop_size])
        else:
            idx = (
                slice(0, 1),
                *[slice(s // 10, s // 10 + int(obj_scale * 4)) for s in crop_size],
            )

        data[idx] = 1
        data[0, 0:3] = 0
        assert data.max() == 1

        boxes, _ = instances_to_boxes_np(data, dim=data.ndim - 1)
        case = {"data": data}

        _fn, _fn_key = BoxEnsemblerSelective.constructor(
            parameters={"model_iou": 0.0000001},
        )
        predictor = Predictor(
            ensembler={_fn_key: _fn},
            models=[DummyBoxModel()],
            crop_size=crop_size,
            device="cpu",
        )
        prediction = predictor.predict_case(case=case, properties=properties_simple)

        assert "boxes" in prediction
        assert "pred_boxes" in prediction["boxes"]
        assert not prediction["boxes"]["restore"]

        # Needs to be changed after https://github.com/MIC-DKFZ/nnDetection/issues/23
        boxes[boxes < 0] = 0
        assert np.allclose(boxes, prediction["boxes"]["pred_boxes"])
        assert np.allclose(np.array([1.0]), prediction["boxes"]["pred_scores"])
        assert np.allclose(np.array([1]), prediction["boxes"]["pred_labels"])
