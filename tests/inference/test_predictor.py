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
