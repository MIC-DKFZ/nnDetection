"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Hashable, Union

import torch
import numpy as np
from scipy.stats import norm
from torch import Tensor

from loguru import logger

from nndet.inference.detection.model import batched_weighted_nms_model
from nndet.inference.detection import batched_nms_model, batched_nms_ensemble, \
    batched_wbc_ensemble, wbc_nms_no_label_ensemble
from nndet.inference.ensembler.base import BaseEnsembler, OverlapMap
from nndet.inference.restore import restore_detection
from nndet.core.boxes import box_center, clip_boxes_to_image, remove_small_boxes
from nndet.utils.tensor import cat, to_device, to_dtype


class BoxEnsembler(BaseEnsembler):
    ID = "boxes"

    def __init__(self,
                 properties: Dict[str, Any],
                 parameters: Dict[str, Any],
                 box_key: str = 'pred_boxes',
                 score_key: str = 'pred_scores',
                 label_key: str = 'pred_labels',
                 data_key: str = 'data',
                 device: Optional[Union[torch.device, str]] = None,
                 **kwargs):
        """
        Ensemble bounding box detections from tta and multiple models

        Args:
            properties: properties of the patient/case (e.g. tranpose axes)
            parameters: parameters for ensembling
            box_key: key where boxes are located inside prediction dict
            score_key: key where scores are located inside prediction dict
            label_key: key where labels are located inside prediction dict
            data_key: key where data is located inside batch dict
            device: device to use for internal computations
            kwargs: passed to super class
        """
        super().__init__(
            properties=properties,
            parameters=parameters,
            device=device,
            **kwargs,
            )
        # parameters to access information from predictions and batches
        self.data_key = data_key
        self.score_key = score_key
        self.label_key = label_key
        self.box_key = box_key
        self.overlap_map = OverlapMap(tuple(self.properties["shape"]))

    @classmethod
    def from_case(cls,
                  case: Dict,
                  properties: Dict,
                  parameters: Optional[Dict] = None,
                  box_key: str = 'pred_boxes',
                  score_key: str = 'pred_scores',
                  label_key: str = 'pred_labels',
                  data_key: str = 'data',
                  device: Optional[Union[torch.device, str]] = None,
                  **kwargs,
                  ):
        """
        Primary way to instantiate this class. Automatically extracts all
        properties and uses a default set of parameters for ensembling.

        Args:
            case: case which is predicted.
            properties: Additional properties.
                Required keys:
                    `transpose_backward`
                    `spacing_after_resampling`
                    `crop_bbox`
            parameters: Additional parameters. Defaults to None.
            box_key: key where boxes are located inside prediction dict
            score_key: key where scores are located inside prediction dict
            label_key: key where labels are located inside prediction dict
            data_key: key where data is located inside batch dict
            device: device to use for internal computations
        """
        _parameters = cls.get_default_parameters()
        _parameters.update(parameters)

        _properties = {
            "shape": case[data_key].shape[1:],  # remove channel dim
            "transpose_backward": properties["transpose_backward"],
            "original_spacing": properties["original_spacing"],
            "spacing_after_resampling": properties["spacing_after_resampling"],
            "crop_bbox": properties["crop_bbox"],
            "original_size_of_raw_data": properties["original_size_of_raw_data"],
            "itk_origin": properties["itk_origin"],
            "itk_spacing": properties["itk_spacing"],
            "itk_direction": properties["itk_direction"],
        }
        return cls(
            properties=_properties,
            parameters=_parameters,
            box_key=box_key,
            score_key=score_key,
            label_key=label_key,
            data_key=data_key,
            device=device,
            **kwargs,
        )

    @classmethod
    def get_default_parameters(cls):
        """
        Generate default parameters for instantiation

        Returns:
            Dict:
                `model_iou`: IoU for model nms function
                `model_nms_fn`: function to use for model NMS
                `model_topk`: number of predictions with the highest
                    probability to keep
                `ensemble_iou`: IoU for ensembling the predictions of multiple
                    models
                `ensemble_nms_fn`: ensemble predictions from multiple
                    models
                `ensemble_nms_topk`: number of predictions with the highest
                    probability to keep
                `ensemble_remove_small_boxes`: minimum size of the box
                `ensemble_score_thresh`: minimum probability
        """
        return {
            # single model
            "model_iou": 0.1,
            "model_nms_fn": batched_nms_model,
            "model_score_thresh": 0.0,
            "model_topk": 1000,
            "model_detections_per_image": 100,

            # ensemble multiple models
            "ensemble_iou": 0.5,
            "ensemble_nms_fn": batched_wbc_ensemble,
            "ensemble_topk": 1000,
            "remove_small_boxes": 1e-2,
            "ensemble_score_thresh": 0.0,
        }

    def postprocess_image(self,
                          boxes: torch.Tensor, 
                          probs: torch.Tensor, 
                          labels: torch.Tensor, 
                          weights: torch.Tensor,
                          shape: Optional[Tuple[int]] = None
                          ) -> Tuple[torch.Tensor, torch.Tensor,
                                     torch.Tensor, torch.Tensor]:
        """
        Postprocessing of a single image
        select topk predictions -> score threshold -> clipping -> \
            remove small boxes -> nms

        Args:
            boxes: predicted deltas for proposals [N, dim * 2]
            probs: predicted logits for boxes [N]
            labels: predicted labels for boxes [N]
            weights: weight for each box [N]

        Returns:
            torch.Tensor: postprocessed boxes
            torch.Tensor: postprocessed probs
            torch.Tensor: postprocessed labels
            torch.Tensor: postprocessed weights
        """
        p_sorted, idx_sorted = probs.sort(descending=True)
        idx_sorted = idx_sorted[:self.parameters["model_topk"]]
        p_sorted = p_sorted[:self.parameters["model_topk"]]
        keep_idxs = p_sorted > self.parameters["model_score_thresh"]
        idx_sorted = idx_sorted[keep_idxs]

        b, p, l, w = boxes[idx_sorted], probs[idx_sorted], labels[idx_sorted], weights[idx_sorted]

        b = clip_boxes_to_image(b, shape)
        # After clipping we could have boxes with volume 0 which we definitely
        # need to remove because of the IoU computation
        keep = remove_small_boxes(
            b, min_size=self.parameters["remove_small_boxes"])
        b, p, l, w = b[keep], p[keep], l[keep], w[keep]

        _boxes, _probs, _labels, _weights = self.parameters["model_nms_fn"](
            boxes=b, scores=p, labels=l, weights=w,
            iou_thresh=self.parameters["model_iou"],
        )

        # predictions are sorted
        _boxes = _boxes[:self.parameters.get("model_detections_per_image", 1000)]
        _probs = _probs[:self.parameters.get("model_detections_per_image", 1000)]
        _labels = _labels[:self.parameters.get("model_detections_per_image", 1000)]
        _weights = _weights[:self.parameters.get("model_detections_per_image", 1000)]
        return _boxes, _probs, _labels, _weights

    @staticmethod
    def _apply_offsets_to_boxes(boxes: List[Tensor],
                                tile_offset: Sequence[Sequence[int]],
                                ) -> List[Tensor]:
        """
        Apply offset to bounding boxes to position them correctly inside
        the whole case

        Args:
            boxes: predicted boxes [N, dims * 2]
                [x1, y1, x2, y2, (z1, z2))
            tile_offset: defines offset for each tile

        Returns:
            List[Tensor]: bounding boxes with respect to origin of whole case
        """
        offset_boxes = []
        for img_boxes, offset in zip(boxes, tile_offset):
            if img_boxes.nelement() == 0:
                offset_boxes.append(img_boxes)
                continue
            offset = Tensor(offset).to(img_boxes)
            _boxes = img_boxes.clone()

            _boxes[:, 0] += offset[0]
            _boxes[:, 1] += offset[1]
            _boxes[:, 2] += offset[0]
            _boxes[:, 3] += offset[1]

            if img_boxes.shape[1] == 6:
                _boxes[:, 4] += offset[2]
                _boxes[:, 5] += offset[2]

            offset_boxes.append(_boxes)
        return offset_boxes

    def restore_prediction(self, boxes: Tensor):
        """
        Restore predictions in the original image space

        Args:
            boxes: predicted boxes [N, dims * 2] (x1, y1, x2, y2, (z1, z2))

        Returns:
            Tensor: boxes in original image space [N, dims * 2]
                (x1, y1, x2, y2, (z1, z2))
        """
        _old_dtype = boxes.dtype
        boxes_np = restore_detection(
            boxes.detach().cpu().numpy(),
            transpose_backward=self.properties["transpose_backward"],
            original_spacing=self.properties["original_spacing"],
            spacing_after_resampling=self.properties["spacing_after_resampling"],
            crop_bbox=self.properties["crop_bbox"],
        )
        boxes = torch.from_numpy(boxes_np).to(dtype=_old_dtype)
        return boxes

    def save_state(self,
                   target_dir: Path,
                   name: str,
                   **kwargs,
                   ):
        """
        Save case result as pickle file. Identifier of ensembler will
        be added to the name

        Args:
            target_dir: folder to save result to
            name: name of case
        
        Notes:
            The device is not saved inside the checkpoint and everything
            will be loaded on the CPU.
        """
        super().save_state(
            target_dir=target_dir,
            name=name,
            score_key=self.score_key,
            label_key=self.label_key,
            box_key=self.box_key,
            data_key=self.data_key,
            overlap_map=self.overlap_map,
            **kwargs,
        )

    @classmethod
    def from_checkpoint(cls, base_dir: PathLike, case_id: str, **kwargs):
        ckp = torch.load(str(Path(base_dir) / f"{case_id}_{cls.ID}.pt"))

        t = cls(
            properties=ckp["properties"],
            parameters=ckp["parameters"],
            box_key=ckp["box_key"],
            score_key=ckp["score_key"],
            label_key=ckp["label_key"],
            data_key=ckp["data_key"],
            **kwargs
        )
        t._load(ckp)
        return t

    @classmethod
    def sweep_parameters(cls) -> Tuple[Dict[str, Any],
                                       Dict[str, Sequence[Any]]]:
        # iou_threshs = np.linspace(0.0, 0.8, 9)
        iou_threshs = np.linspace(0.0, 0.5, 6)
        iou_threshs[0] = 1e-5
        small_boxes_thresh = np.linspace(2., 7., 6)

        param_sweep = {
            # ensemble multiple models
            "ensemble_iou": iou_threshs,
            "model_score_thresh": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            # "remove_small_boxes": small_boxes_thresh,
        }
        return cls.get_default_parameters(), param_sweep

    @torch.no_grad()
    def process_batch(self, result: Dict, batch: Dict):
        """
        Process a single batch of bounding box predictions
        (the boxes are clipped to the case size in the ensembling step)

        Args:
            result: prediction from detector. Need to provide boxes, scores
                and class labels
                    `self.box_key`: List[Tensor]: predicted boxes (relative
                        to patch coordinates)
                    `self.score_key` List[Tensor]: score for each tensor
                    `self.label_key`: List[Tensor] label prediction for each box
            batch: input batch for detector
                `tile_origin: origin of crop with respect to actual data (
                    in case of padding)
                `crop`: Sequence[slice] original crop from data
        
        Warnings:
            Make sure to move cached values to the CPU after they have been
            processed.
        """
        tile_origins = [to for to in zip(*batch["tile_origin"])]
        tile_size = batch[self.data_key].shape[2:]

        boxes = []
        scores = []
        labels = []
        for b, s, l in zip(result[self.box_key], result[self.score_key], result[self.label_key]):
            _boxes, _scores, _labels, _ = self.postprocess_image(
                boxes=b.float(),
                probs=s.float(),
                labels=l.float(),
                weights=torch.ones_like(s).float(),
                shape=tuple(tile_size),
                )
            boxes.append(_boxes.cpu())
            scores.append(_scores.cpu())
            labels.append(_labels.cpu())

        centers = [box_center(img_boxes) if img_boxes.numel() > 0 else Tensor([]).to(img_boxes)
                   for img_boxes in boxes]
        weights = [self._get_box_in_tile_weight(c, tile_size) for c in centers]
        weights = [w * self.model_weights[self.model_current] for w in weights]

        boxes = self._apply_offsets_to_boxes(boxes, tile_origins)

        self.model_results[self.model_current]["boxes"].extend(boxes)
        self.model_results[self.model_current]["scores"].extend(scores)
        self.model_results[self.model_current]["labels"].extend(labels)
        self.model_results[self.model_current]["weights"].extend(weights)

        crops_reshaped = list(zip(*batch["crop"]))
        self.model_results[self.model_current]["crops"].extend(crops_reshaped)

        for crop in crops_reshaped:
            self.overlap_map.add_overlap(crop)

    @staticmethod
    def _get_box_in_tile_weight(box_centers: Tensor,
                                tile_size: Sequence[int],
                                ) -> Tensor:
        """
        Assign boxes at the corners of tiles a lower weight (weight
        is drawn form a scaled normal distribution)

        Args:
            box_centers: center predicted box [N, dims]
            tile_size: size the of patch/tile

        Returns:
            Tensor: weight for each bounding box [N]
        """
        if box_centers.numel() > 0:
            all_weights = []
            centers_np = box_centers.detach().cpu().numpy()
            for center_np in centers_np:
                weight = np.mean([
                    norm.pdf(bc, loc=ps, scale=ps * 0.8) * np.sqrt(2 * np.pi) * ps * 0.8
                    for bc, ps in zip(center_np, np.array(tile_size) / 2)])
                all_weights.append([weight])
            return torch.from_numpy(np.concatenate(all_weights)).to(box_centers)
        else:
            return Tensor([]).to(box_centers)

    @torch.no_grad()
    def get_case_result(self,
                        restore: bool = False,
                        names: Optional[Sequence[Hashable]] = None,
                        ) -> Dict[str, Tensor]:
        """
        Process all the batches and models and create the final prediction

        Args:
            restore: restore prediction in the original image space
            names: name of the models to use. By default all models are used.

        Returns:
            Dict: final result
                `pred_boxes`: predicted box locations
                    [N, dims * 2] (x1, y1, x2, y2, (z1, z2))
                `pred_scores`: predicted probability per box [N]
                `pred_labels`: predicted label per box [N]
                `restore`: indicate whether predictions were restored in
                    original image space
                `original_size_of_raw_data`: image shape befor preprocessing
                `itk_origin`: itk origin of image before preprocessing
                `itk_spacing`: itk spacing of image before preprocessing
                `itk_direction`: itk direction of image before preprocessing
        """
        if names is None:
            names = list(self.model_results.keys())

        boxes, probs, labels, weights = [], [], [], []
        for name in names:
            _boxes, _probs, _labels, _weights = self.process_model(name)
            boxes.append(_boxes)
            probs.append(_probs)
            labels.append(_labels)
            weights.append(_weights)

        boxes, probs, labels = self.process_ensemble(
            boxes=boxes, probs=probs, labels=labels,
            weights=weights,
        )

        if restore:
            boxes = self.restore_prediction(boxes)

        return {
            "pred_boxes": boxes,
            "pred_scores": probs,
            "pred_labels": labels,
            "restore": restore,
            "original_size_of_raw_data": self.properties["original_size_of_raw_data"],
            "itk_origin": self.properties["itk_origin"],
            "itk_spacing": self.properties["itk_spacing"],
            "itk_direction": self.properties["itk_direction"],
            }

    def process_model(self, name: Hashable) ->\
            Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Process the output of a single model on the whole scan
        topk candidates -> nms

        Args:
            name: name of model to process

        Returns:
            Tensor: filtered boxes
            Tensor: filtered probs
            Tensor: filtered labels
            idx: indices kept from original ordered data
        """
        # concatenate batches
        boxes = cat(self.model_results[name]["boxes"], dim=0)
        probs = cat(self.model_results[name]["scores"], dim=0)
        labels = cat(self.model_results[name]["labels"], dim=0)
        weights = cat(self.model_results[name]["weights"], dim=0)
        return boxes, probs, labels, weights

    def process_ensemble(self, boxes: List[Tensor], probs: List[Tensor],
                         labels: List[Tensor], weights: List[Tensor],
                         ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Ensemble predictions from multiple models

        Args:
            boxes: predicted boxes List[[N, dims * 2]]
                (x1, y1, x2, y2, (z1, z2))
            probs: predicted probabilities List[[N]]
            labels: predicted label List[[N]]
            weights: additional weight List[[N]]

        Returns:
            Tensor: ensembled box predictions
            Tensor: ensembled probabilities
            Tensor: ensembled labels
        """
        boxes = cat(boxes, dim=0)
        probs = cat(probs, dim=0)
        labels = cat(labels, dim=0)
        weights = cat(weights, dim=0)

        _, idx = probs.sort(descending=True)
        idx = idx[:self.parameters["ensemble_topk"]]
        boxes = boxes[idx]
        probs = probs[idx]
        labels = labels[idx]
        weights = weights[idx]

        n_exp_preds = self.overlap_map.mean_num_overlap_of_boxes(boxes)
        boxes, probs, labels = self.parameters["ensemble_nms_fn"](
            boxes, probs, labels,
            weights=weights,
            iou_thresh=self.parameters["model_iou"],
            n_exp_preds=n_exp_preds,
            score_thresh=self.parameters["ensemble_score_thresh"],
        )
        return boxes.cpu(), probs.cpu(), labels.cpu()


class BoxEnsemblerLW(BoxEnsembler):
    """
    Uses different computation for box weight, much faster than box ensembler.
    """
    @staticmethod
    def _get_box_in_tile_weight(box_centers: Tensor,
                                tile_size: Sequence[int],
                                ) -> Tensor:
        """
        Assign boxes near the corner a lower weight.
        The middle has a plateau with weight one, starting from patchsize / 2
        the weights decreases linearly until 0.5 is reached. 

        Args:
            box_centers: center predicted box [N, dims]
            tile_size: size the of patch/tile

        Returns:
            Tensor: weight for each bounding box [N]
        """
        plateau_length = 0.5 # adjust width of plateau and min weight
        if box_centers.numel() > 0:
            tile_center = torch.tensor(tile_size).to(box_centers) / 2.  # [dims]

            max_dist = tile_center.norm(p=2)  # [1]
            boxes_dist = (box_centers - tile_center[None]).norm(p=2, dim=1)  # [N]
            weight = -(boxes_dist / max_dist - plateau_length).clamp_(min=0) + 1
            return weight
        else:
            return Tensor([]).to(box_centers)


class BoxEnsemblerFastest(BoxEnsemblerLW):
    """
    Uses the fastest but not necessarily most precise box ensembling strategy

    Only save top `num_reduced_cache` boxes for ensembling
    Uses a linear box weight
    Uses the mean over the whole overlap map. Depending on overlap
    and patch stride this is not correct.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reduced_cache = False
        self.num_reduced_cache = 8000
        self.overlap_map_mean = None

    @classmethod
    def get_default_parameters(cls):
        """
        Generate default parameters for instantiation

        Returns:
            Dict:
                `model_iou`: IoU for model nms function
                `model_nms_fn`: function to use for model NMS
                `model_topk`: number of predictions with the highest
                    probability to keep
                `ensemble_iou`: IoU for ensembling the predictions of multiple
                    models
                `ensemble_nms_fn`: ensemble predictions from multiple
                    models
                `ensemble_nms_topk`: number of predictions with the highest
                    probability to keep
                `ensemble_remove_small_boxes`: minimum size of the box
                `ensemble_score_thresh`: minimum probability
        """
        return {
            # single model
            "model_iou": 0.1,
            "model_nms_fn": batched_nms_model,
            "model_score_thresh": 0.1,
            "model_topk": 1000,
            "model_detections_per_image": 1000,

            # ensemble multiple models
            "ensemble_iou": 0.5,
            "ensemble_nms_fn": batched_wbc_ensemble,
            "ensemble_topk": 1000,
            "remove_small_boxes": 1e-2,
            "ensemble_score_thresh": 0.0,
        }

    @classmethod
    def sweep_parameters(cls) -> Tuple[Dict[str, Any],
                                       Dict[str, Sequence[Any]]]:
        iou_threshs = np.linspace(0.0, 0.5, 6)
        iou_threshs[0] = 1e-5
        small_boxes_thresh = [1e-2] + np.linspace(2., 7., 6).tolist()

        param_sweep = {
            # single model
            "model_iou": iou_threshs,
            # ensemble multiple models
            "ensemble_iou": iou_threshs,
            "remove_small_boxes": small_boxes_thresh,
            "model_score_thresh": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
        return cls.get_default_parameters(), param_sweep

    @torch.no_grad()
    def process_batch(self, result: Dict, batch: Dict):
        """
        Process a single batch of bounding box predictions
        (the boxes are clipped to the case size in the ensembling step)

        Args:
            result: prediction from detector. Need to provide boxes, scores
                and class labels
                    `self.box_key`: List[Tensor]: predicted boxes (relative
                        to patch coordinates)
                    `self.score_key` List[Tensor]: score for each tensor
                    `self.label_key`: List[Tensor] label prediction for each box
            batch: input batch for detector
                `tile_origin: origin of crop with respect to actual data (
                    in case of padding)
                `crop`: Sequence[slice] original crop from data
        """
        if self.reduced_cache:
            logger.warning("Ensembler was already reduced, need to rerun reduce_cache "
                           "later and restore overlap map with proxy mean.")
            self.overlap_map.restore_mean(self.overlap_map_mean)
            self.reduced_cache = False

        boxes = [r.half().cpu() for r in result[self.box_key]]
        scores = [r.half().cpu() for r in result[self.score_key]]
        labels = [r.half().cpu() for r in result[self.label_key]]
        centers = [box_center(img_boxes) if img_boxes.numel() > 0 else Tensor([]).to(img_boxes)
                   for img_boxes in boxes]
        tile_origins = [to for to in zip(*batch["tile_origin"])]

        tile_size = batch[self.data_key].shape[2:]
        weights = [self._get_box_in_tile_weight(c, tile_size) for c in centers]
        weights = [w * self.model_weights[self.model_current] for w in weights]

        boxes = self._apply_offsets_to_boxes(boxes, tile_origins)

        self.model_results[self.model_current]["boxes"].extend(boxes)
        self.model_results[self.model_current]["scores"].extend(scores)
        self.model_results[self.model_current]["labels"].extend(labels)
        self.model_results[self.model_current]["weights"].extend(weights)

        crops_reshaped = list(zip(*batch["crop"]))
        self.model_results[self.model_current]["crops"].extend(crops_reshaped)

        for crop in crops_reshaped:
            self.overlap_map.add_overlap(crop)

    @staticmethod
    def _get_box_in_tile_weight(box_centers: Tensor,
                                tile_size: Sequence[int],
                                ) -> Tensor:
        """
        Assign boxes near the corner a lower weight.
        The middle has a plateau with weight one, starting from patchsize / 2
        the weights decreases linearly until 0.5 is reached.

        Args:
            box_centers: center predicted box [N, dims]
            tile_size: size the of patch/tile

        Returns:
            Tensor: weight for each bounding box [N]
        """
        plateau_length = 0.5 # adjust width of plateau and min weight
        if box_centers.numel() > 0:
            tile_center = torch.tensor(tile_size).to(box_centers) / 2.  # [dims]

            max_dist = tile_center.norm(p=2)  # [1]
            boxes_dist = (box_centers - tile_center[None]).norm(p=2, dim=1)  # [N]
            weight = -(boxes_dist / max_dist - plateau_length).float().clamp_(min=0).half() + 1
            return weight
        else:
            return Tensor([]).to(box_centers).half()

    def process_model(self,
                      name: Hashable,
                      ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Process the output of a single model on the whole scan
        topk candidates -> nms

        Args:
            name: name of model to process

        Returns:
            Tensor: processed boxes
            Tensor: processed probs
            Tensor: processed labels
            Tensor: processed weights
        """
        boxes = to_device(self.model_results[name]["boxes"], device=self.device)
        probs = to_device(self.model_results[name]["scores"], device=self.device)
        labels = to_device(self.model_results[name]["labels"], device=self.device)
        weights = to_device(self.model_results[name]["weights"], device=self.device)

        model_boxes = []
        model_probs = []
        model_labels = []
        model_weights = []
        for b, p, l, w in zip(boxes, probs, labels, weights):
            if b.numel() > 0:
                _b, _p, _l, _w = self.postprocess_image(
                    boxes=b.float(),
                    probs=p.float(),
                    labels=l.float(),
                    weights=w.float(),
                    shape=tuple(self.properties["shape"]),
                    )
                model_boxes.append(_b)
                model_probs.append(_p)
                model_labels.append(_l)
                model_weights.append(_w)
        return cat(model_boxes), cat(model_probs), cat(model_labels), cat(model_weights)

    def process_ensemble(self,
                         boxes: List[Tensor],
                         probs: List[Tensor],
                         labels: List[Tensor],
                         weights: List[Tensor],
                         ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Ensemble predictions from multiple models

        Args:
            boxes: predicted boxes List[[N, dims * 2]]
                (x1, y1, x2, y2, (z1, z2))
            probs: predicted probabilities List[[N]]
            labels: predicted label List[[N]]
            weights: additional weight List[[N]]

        Returns:
            Tensor: ensembled box predictions
            Tensor: ensembled probabilities
            Tensor: ensembled labels
        """
        boxes = cat(boxes, dim=0)
        probs = cat(probs, dim=0)
        labels = cat(labels, dim=0)
        weights = cat(weights, dim=0)

        _, idx = probs.sort(descending=True)
        idx = idx[:self.parameters["ensemble_topk"]]
        boxes = boxes[idx]
        probs = probs[idx]
        labels = labels[idx]
        weights = weights[idx]

        n_exp_preds = self.overlap_map_mean.expand(len(boxes)).to(boxes)
        boxes, probs, labels = self.parameters["ensemble_nms_fn"](
            boxes, probs, labels,
            weights=weights,
            iou_thresh=self.parameters["model_iou"],
            n_exp_preds=n_exp_preds,
            score_thresh=self.parameters["ensemble_score_thresh"],
        )
        return boxes.cpu(), probs.cpu(), labels.cpu()

    @torch.no_grad()
    def get_case_result(self,
                        restore: bool = False,
                        names: Optional[Sequence[Hashable]] = None,
                        ) -> Dict[str, Tensor]:
        """
        Process all the batches and models and create the final prediction

        Args:
            restore: restore prediction in the original image space
            names: name of the models to use. By default all models are used.

        Returns:
            Dict: final result
                `pred_boxes`: predicted box locations
                    [N, dims * 2] (x1, y1, x2, y2, (z1, z2))
                `pred_scores`: predicted probability per box [N]
                `pred_labels`: predicted label per box [N]
                `restore`: indicate whether predictions were restored in
                    original image space
                `original_size_of_raw_data`: image shape befor preprocessing
                `itk_origin`: itk origin of image before preprocessing
                `itk_spacing`: itk spacing of image before preprocessing
                `itk_direction`: itk direction of image before preprocessing
        """
        self.reduce_cache()
        return super().get_case_result(restore=restore, names=names)

    def save_state(self,
                   target_dir: Path,
                   name: str,
                   **kwargs,
                   ):
        """
        Save case result as pickle file. Identifier of ensembler will
        be added to the name. Before saving the state, the cache will
        be reduced to a predefined number of predictions to for memory
        and computational reasons

        Args:
            target_dir: folder to save result to
            name: name of case
        
        Notes:
            The device is not saved inside the checkpoint and everything
            will be loaded on the CPU.
        """
        self.reduce_cache()
        return BaseEnsembler.save_state(
            self,
            target_dir=target_dir,
            name=name,
            reduced_cache=self.reduced_cache,
            score_key=self.score_key,
            label_key=self.label_key,
            box_key=self.box_key,
            data_key=self.data_key,
            overlap_map_mean=self.overlap_map_mean,
            **kwargs,
            )

    def reduce_cache(self):
        """
        Only save a subset of all boxes for further evaluations
        """
        if not self.reduced_cache:
            self.reduced_cache = True
            # we use the mean here to save time ...
            self.overlap_map_mean = self.overlap_map.avg()

            for model in self.model_results.keys():
                batch_idx = self.build_batch_indices(self.model_results[model]["scores"])

                boxes = cat(self.model_results[model]["boxes"])
                probs = cat(self.model_results[model]["scores"])
                labels = cat(self.model_results[model]["labels"])
                weights = cat(self.model_results[model]["weights"])

                if len(probs) > self.num_reduced_cache:
                    _, idx_sorted = probs.sort(descending=True)
                    idx_sorted = idx_sorted[:self.num_reduced_cache]
                    batch_idx_keep = [[b for b in bix if b in idx_sorted] for bix in batch_idx]

                    assert len(batch_idx_keep) == len(self.model_results[model]["scores"])

                    self.model_results[model]["boxes"] = [boxes[i] for i in batch_idx_keep]
                    self.model_results[model]["scores"] = [probs[i] for i in batch_idx_keep]
                    self.model_results[model]["labels"] = [labels[i] for i in batch_idx_keep]
                    self.model_results[model]["weights"] = [weights[i] for i in batch_idx_keep]

    @staticmethod
    def build_batch_indices(b: Sequence[Tensor]) -> List[List[int]]:
        idx = []
        num_elem = 0
        for _b in b:
            if _b.numel() > 0:
                additional_elem = len(_b)
                idx.append(list(range(num_elem, num_elem + additional_elem)))
                num_elem += additional_elem
            else:
                idx.append([])
        return idx


class BoxEnsemblerSelective(BoxEnsembler):
    def __init__(self,
                 properties: Dict[str, Any],
                 parameters: Dict[str, Any],
                 box_key: str = 'pred_boxes',
                 score_key: str = 'pred_scores',
                 label_key: str = 'pred_labels',
                 data_key: str = 'data',
                 device: Optional[Union[torch.device, str]] = None,
                 **kwargs,
                 ):
        """
        Ensemble bounding box detections from tta and multiple models
        This uses a different ensembling strategy which is faster and allows
        for model IoU optimization.

        Args:
            properties: properties of the patient/case (e.g. tranpose axes)
            parameters: parameters for ensembling
            box_key: key where boxes are located inside prediction dict
            score_key: key where scores are located inside prediction dict
            label_key: key where labels are located inside prediction dict
            data_key: key where data is located inside batch dict
            device: device to use for internal computations
            kwargs: passed to super class
        """
        super().__init__(
            properties=properties,
            parameters=parameters,
            device=device,
            box_key=box_key,
            score_key=score_key,
            label_key=label_key,
            data_key=data_key,
            **kwargs,
            )
        self.overlap_map = None

    @classmethod
    def get_default_parameters(cls):
        """
        Generate default parameters for instantiation

        Returns:
            Dict:
                `model_iou`: IoU for model nms function
                `model_nms_fn`: function to use for model NMS
                `model_topk`: number of predictions with the highest
                    probability to keep
                `ensemble_iou`: IoU for ensembling the predictions of multiple
                    models
                `ensemble_nms_fn`: ensemble predictions from multiple
                    models
                `ensemble_nms_topk`: number of predictions with the highest
                    probability to keep
                `ensemble_remove_small_boxes`: minimum size of the box
                `ensemble_score_thresh`: minimum probability
        """
        return {
            # single model
            "model_iou": 0.1,
            "model_nms_fn": batched_weighted_nms_model,
            "model_score_thresh": 0.0,
            "model_topk": 1000,
            "model_detections_per_image": 100,

            # ensemble multiple models
            "ensemble_iou": 0.5,
            "ensemble_nms_fn": batched_wbc_ensemble,
            "ensemble_topk": 1000,
            "remove_small_boxes": 1e-2,
            "ensemble_score_thresh": 0.0,
        }

    @classmethod
    def sweep_parameters(cls) -> Tuple[Dict[str, Any],
                                       Dict[str, Sequence[Any]]]:
        # iou_threshs = np.linspace(0.0, 0.8, 9)
        iou_threshs = np.linspace(0.0, 0.5, 6)
        iou_threshs[0] = 1e-5
        small_boxes_thresh = [1e-2] + np.linspace(2., 7., 6).tolist()

        param_sweep = {
            # single model
            "model_iou": iou_threshs,
            "model_nms_fn": [
                batched_weighted_nms_model,
                batched_nms_model,
            ],
            # ensemble multiple models
            "ensemble_iou": iou_threshs,
            "model_score_thresh": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "remove_small_boxes": small_boxes_thresh,
        }
        return cls.get_default_parameters(), param_sweep

    @torch.no_grad()
    def process_batch(self, result: Dict, batch: Dict):
        """
        Process a single batch of bounding box predictions
        (the boxes are clipped to the case size in the ensembling step)

        Args:
            result: prediction from detector. Need to provide boxes, scores
                and class labels
                    `self.box_key`: List[Tensor]: predicted boxes (relative
                        to patch coordinates)
                    `self.score_key` List[Tensor]: score for each tensor
                    `self.label_key`: List[Tensor] label prediction for each box
            batch: input batch for detector
                `tile_origin: origin of crop with respect to actual data (
                    in case of padding)
                `crop`: Sequence[slice] original crop from data
        """
        boxes = [r.float().cpu() for r in result[self.box_key]]
        scores = [r.float().cpu() for r in result[self.score_key]]
        labels = [r.float().cpu() for r in result[self.label_key]]
        centers = [box_center(img_boxes) if img_boxes.numel() > 0 else Tensor([]).to(img_boxes)
                   for img_boxes in boxes]
        tile_origins = [to for to in zip(*batch["tile_origin"])]

        tile_size = batch[self.data_key].shape[2:]
        weights = [self._get_box_in_tile_weight(c, tile_size) for c in centers]
        weights = [w * self.model_weights[self.model_current] for w in weights]

        boxes = self._apply_offsets_to_boxes(boxes, tile_origins)

        self.model_results[self.model_current]["boxes"].extend(boxes)
        self.model_results[self.model_current]["scores"].extend(scores)
        self.model_results[self.model_current]["labels"].extend(labels)
        self.model_results[self.model_current]["weights"].extend(weights)
        # self.model_results[self.model_current]["crops"].extend(
        #     list(zip(*batch["crop"])))

    @staticmethod
    def _get_box_in_tile_weight(box_centers: Tensor,
                                tile_size: Sequence[int],
                                ) -> Tensor:
        """
        Assign boxes near the corner a lower weight.
        The midle has a plateau with weight one, starting from patchsize / 2
        the weights decreases linearly until 0.5 is reached. 

        Args:
            box_centers: center predicted box [N, dims]
            tile_size: size the of patch/tile

        Returns:
            Tensor: weight for each bounding box [N]
        """
        plateau_length = 0.5  # adjust width of plateau and min weight
        if box_centers.numel() > 0:
            tile_center = torch.tensor(tile_size).to(box_centers) / 2.  # [dims]

            max_dist = tile_center.norm(p=2)  # [1]
            boxes_dist = (box_centers - tile_center[None]).norm(p=2, dim=1) # [N]
            weight = -(boxes_dist / max_dist - plateau_length).clamp_(min=0) + 1
            return weight
        else:
            return Tensor([]).to(box_centers)

    def process_model(self, name: Hashable) ->\
            Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Process the output of a single model on the whole scan
        topk candidates -> nms

        Args:
            name: name of model to process

        Returns:
            Tensor: processed boxes
            Tensor: processed probs
            Tensor: processed labels
            Tensor: processed weights
        """
        # collect predictions on whole case and apply postprocessing
        boxes = cat(self.model_results[name]["boxes"]).to(self.device)
        probs = cat(self.model_results[name]["scores"]).to(self.device)
        labels = cat(self.model_results[name]["labels"]).to(self.device)
        weights = cat(self.model_results[name]["weights"]).to(self.device)

        return self.postprocess_image(
            boxes=boxes,
            probs=probs,
            labels=labels,
            weights=weights,
            shape=tuple(self.properties["shape"]),
            )

    def process_ensemble(self, boxes: List[Tensor], probs: List[Tensor],
                         labels: List[Tensor], weights: List[Tensor],
                         ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Ensemble predictions from multiple models

        Args:
            boxes: predicted boxes List[[N, dims * 2]]
                (x1, y1, x2, y2, (z1, z2))
            probs: predicted probabilities List[[N]]
            labels: predicted label List[[N]]
            weights: additional weight List[[N]]

        Returns:
            Tensor: ensembled box predictions
            Tensor: ensembled probabilities
            Tensor: ensembled labels
        """
        num_models = len(boxes)
        boxes = cat(boxes, dim=0)
        probs = cat(probs, dim=0)
        labels = cat(labels, dim=0)
        weights = cat(weights, dim=0)

        _, idx = probs.sort(descending=True)
        idx = idx[:self.parameters["ensemble_topk"]]
        boxes = boxes[idx]
        probs = probs[idx]
        labels = labels[idx]
        weights = weights[idx]

        n_exp_preds = torch.tensor([num_models] * len(boxes)).to(boxes)
        boxes, probs, labels = self.parameters["ensemble_nms_fn"](
            boxes, probs, labels,
            weights=weights,
            iou_thresh=self.parameters["ensemble_iou"],
            n_exp_preds=n_exp_preds,
            score_thresh=self.parameters["ensemble_score_thresh"],
        )
        return boxes.cpu(), probs.cpu(), labels.cpu()

    def save_state(self,
                   target_dir: Path,
                   name: str,
                   **kwargs,
                   ):
        """
        Save case result as pickle file. Identifier of ensembler will
        be added to the name.
        This version only saves the topk model predictions to speed
        up loading.

        Args:
            target_dir: folder to save result to
            name: name of case
        
        Notes:
            The device is not saved inside the checkpoint and everything
            will be loaded on the CPU.
        """
        for model in self.model_results.keys():
            boxes = cat(self.model_results[model]["boxes"])
            probs = cat(self.model_results[model]["scores"])
            labels = cat(self.model_results[model]["labels"])
            weights = cat(self.model_results[model]["weights"])

            if len(probs) > self.parameters["model_topk"]:
                _, idx_sorted = probs.sort(descending=True)
                idx_sorted = idx_sorted[:self.parameters["model_topk"]]            
                self.model_results[model]["boxes"] = boxes[idx_sorted]
                self.model_results[model]["scores"] = probs[idx_sorted]
                self.model_results[model]["labels"] = labels[idx_sorted]
                self.model_results[model]["weights"] = weights[idx_sorted]

        return super().save_state(target_dir=target_dir, name=name, **kwargs)
