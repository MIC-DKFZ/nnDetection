# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

import nndet.core.ops_torch as ops_torch
from nndet.core.boxes.nms import batched_nms, batched_weighted_nms
from nndet.inference.ensembler.base import BaseEnsembler, OverlapMap
from nndet.inference.ensembler.utils import (
    apply_offsets_to_boxes,
    get_box_in_tile_weight_linear,
)
from nndet.inference.restore import restore_boxes
from nndet.io import save_pickle
from nndet.utils.tensor import cat, to_numpy


class MaskEnsembler(BaseEnsembler):
    ID = "masks"

    def __init__(
        self,
        properties: Dict[str, Any],
        parameters: Dict[str, Any],
        mask_key: str = "pred_masks",
        score_key: str = "pred_mask_scores",
        label_key: str = "pred_mask_labels",
        data_key: str = "data",
        box_key: Optional[str] = "pred_boxes",
        device: Optional[Union[torch.device, str]] = None,
        **kwargs,
    ):
        """
        Ensemble bounding box detections from tta and multiple models

        Args:
            properties: properties of the patient/case (e.g. tranpose axes)
            parameters: parameters for ensembling
            mask_key: key where masks are located inside prediction dict
            score_key: key where scores are located inside prediction dict
            label_key: key where labels are located inside prediction dict
            data_key: key where data is located inside batch dict
            box_key: key where boxes are located inside prediction dict
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
        self.mask_key = mask_key
        self.score_key = score_key
        self.label_key = label_key
        self.box_key = box_key
        self.overlap_map = OverlapMap(tuple(self.properties["shape"]))

    @classmethod
    def constructor(
        cls,
        parameters: Optional[Dict] = None,
        mask_key: str = "pred_masks",
        score_key: str = "pred_mask_scores",
        label_key: str = "pred_mask_labels",
        data_key: str = "data",
        box_key: Optional[str] = "pred_boxes",
        device: Optional[Union[torch.device, str]] = None,
        **kwargs,
    ) -> Callable[[Dict, Dict], BaseEnsembler]:
        """
        Get a contructor for this class. Automatically extracts all
        properties and uses a default set of parameters for ensembling.

        Args:
            parameters: Additional parameters. Defaults to None.
            mask_key: key where masks are located inside prediction dict
            score_key: key where scores are located inside prediction dict
            label_key: key where labels are located inside prediction dict
            data_key: key where data is located inside batch dict
            box_key: key where boxes are located inside prediction dictc
            device: device to use for internal computations

        Returns:
            Callable: callable to isntantiate ensembler class with two
                input variable:
                    `case`: input data from case (e.g. 'data' to extract shape
                        information)
                    `properties`: additional properties of case
                        Required keys:
                            `transpose_backward`
                            `spacing_after_resampling`
                            `crop_bbox`
                            `original_size_of_raw_data`
                            `itk_origin`
                            `itk_spacing`
                            `itk_direction`
            str: identifier of ensembler class. This needs to be used as the
                key when construction the ensembler dict for the predictor!
        """

        def create(
            case: Dict,
            properties: Dict,
            *args,
            **kwargs2,
        ):
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
                mask_key=mask_key,
                score_key=score_key,
                label_key=label_key,
                data_key=data_key,
                box_key=box_key,
                device=device,
                *args,
                **kwargs,
                **kwargs2,
            )

        return create, cls.ID

    @torch.no_grad()
    def get_case_result(
        self,
        restore: bool = False,
        names: Optional[Sequence[Hashable]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process all the batches and models and create the final prediction

        Args:
            restore: restore prediction in the original image space
            names: name of the models to use. By default all models are used.

        Returns:
            Dict: final result
                `pred_boxes`: predicted box locations
                    [N, dims * 2] (x1, y1, x2, y2, (z1, z2))
                `pred_masks`: predicted masks
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

        boxes, masks, probs, labels, weights = [], [], [], [], []
        for name in names:
            _boxes, _masks, _probs, _labels, _weights = self.process_model(name)
            boxes.append(_boxes)
            masks.append(_masks)
            probs.append(_probs)
            labels.append(_labels)
            weights.append(_weights)

        boxes, masks, probs, labels = self.process_ensemble(
            boxes=boxes,
            masks=masks,
            probs=probs,
            labels=labels,
            weights=weights,
        )

        if restore:
            boxes, masks = self.restore_prediction(boxes, masks)
        else:
            masks = self._roi_mask_to_image_mask(boxes, masks)

        return {
            "pred_boxes": boxes,
            "pred_masks": masks,
            "pred_scores": probs,
            "pred_labels": labels,
            "restore": restore,
            "original_size_of_raw_data": self.properties["original_size_of_raw_data"],
            "itk_origin": self.properties["itk_origin"],
            "itk_spacing": self.properties["itk_spacing"],
            "itk_direction": self.properties["itk_direction"],
        }

    def restore_prediction(self, boxes: torch.Tensor, masks: torch.Tensor):
        """
        Restore predictions in the original image space

        Args:
            boxes: predicted boxes [N, dims * 2] (x1, y1, x2, y2, (z1, z2))
            masks: predicted masks [N, dims]

        Returns:
            Tensor: boxes in original image space [N, dims * 2]
                (x1, y1, x2, y2, (z1, z2))
            Tensor: masks in image space [N, image_dims]
        """
        _old_dtype = boxes.dtype
        boxes_np = restore_boxes(
            boxes.detach().cpu().numpy(),
            transpose_backward=self.properties["transpose_backward"],
            original_spacing=self.properties["original_spacing"],
            spacing_after_resampling=self.properties["spacing_after_resampling"],
            crop_bbox=self.properties["crop_bbox"],
        )
        boxes = torch.from_numpy(boxes_np).to(dtype=_old_dtype)

        transposing = [0] + [i + 1 for i in self.properties["transpose_backward"]]
        masks = np.transpose(masks, transposing)
        image_masks = ops_torch.roi_mask_to_image_mask(
            boxes=boxes,
            masks=masks,
            image_shape=tuple(self.properties["original_size_of_raw_data"]),
            mode=self.interpolated_mode,
            align_corners=self.parameters["align_corners"],
            threshold=self.parameters["bin_mask_threshold"],
        )
        return boxes, image_masks

    def _roi_mask_to_image_mask(self, boxes: torch.Tensor, masks: torch.Tensor):
        """
        Convert roi masks into image space

        Args:
            boxes: predicted boxes [N, dims * 2] (x1, y1, x2, y2, (z1, z2))
            masks: predicted masks [N, dims]

        Returns:
            Tensor: boxes in original image space [N, dims * 2]
                (x1, y1, x2, y2, (z1, z2))
            Tensor: masks in image space [N, image_dims]
        """
        assert masks.ndim == (boxes.shape[1] // 2) + 1, f"Found mask with {masks.ndim} and boxes with {boxes.shape[1]}"
        image_masks = ops_torch.roi_mask_to_image_mask(
            boxes=boxes,
            masks=masks,
            image_shape=tuple(self.properties["shape"]),
            mode=self.interpolated_mode,
            align_corners=self.parameters["align_corners"],
            threshold=self.parameters["bin_mask_threshold"],
        )
        return image_masks

    def save_state(
        self,
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
            mask_key=self.mask_key,
            data_key=self.data_key,
            overlap_map=self.overlap_map,
            **kwargs,
        )

    @classmethod
    def from_checkpoint(cls, base_dir: os.PathLike, case_id: str, **kwargs):
        ckp = torch.load(str(Path(base_dir) / f"{case_id}_{cls.ID}.pt"))

        t = cls(
            properties=ckp["properties"],
            parameters=ckp["parameters"],
            box_key=ckp["box_key"],
            mask_key=ckp["mask_key"],
            score_key=ckp["score_key"],
            label_key=ckp["label_key"],
            data_key=ckp["data_key"],
            **kwargs,
        )
        t._load(ckp)
        return t

    @classmethod
    def save_result(cls, data: Dict, target_dir: Path, case_name: str) -> None:
        # name without extension!
        data_numpy = to_numpy(data)

        boxes_result = {
            "pred_boxes": data_numpy["pred_boxes"],
            "pred_scores": data_numpy["pred_scores"],
            "pred_labels": data_numpy["pred_labels"],
            "restore": data_numpy["restore"],
            "original_size_of_raw_data": data_numpy["original_size_of_raw_data"],
            "itk_origin": data_numpy["itk_origin"],
            "itk_spacing": data_numpy["itk_spacing"],
            "itk_direction": data_numpy["itk_direction"],
        }
        masks_result = {
            "pred_masks": data_numpy["pred_masks"],
            "pred_scores": data_numpy["pred_scores"],
            "pred_labels": data_numpy["pred_labels"],
            "restore": data_numpy["restore"],
        }
        masks_meta = {
            "original_size_of_raw_data": data_numpy["original_size_of_raw_data"],
            "itk_origin": data_numpy["itk_origin"],
            "itk_spacing": data_numpy["itk_spacing"],
            "itk_direction": data_numpy["itk_direction"],
        }

        save_pickle(boxes_result, target_dir / f"{case_name}_boxes.pkl")
        save_pickle(masks_meta, target_dir / f"{case_name}_{cls.ID}.pkl")
        np.savez_compressed(target_dir / f"{case_name}_{cls.ID}.npz", **masks_result)

    @property
    def interpolated_mode(self):
        dim = len(tuple(self.properties["shape"]))

        if self.parameters["interpolation_mode"] == "linear":
            if dim == 2:
                interp = "bilinear"
            elif dim == 3:
                interp = "trilinear"
            else:
                raise RuntimeError(f"Dim {dim} not supported in interpolation mode.")
        else:
            interp = self.parameters["interpolation_mode"]
        return interp


# TODO: IMPORTANT: Mask representation no channel
# TODO: downstream interfaces


class MaskViaBoxesSelectiveEnsembler(MaskEnsembler):
    def __init__(
        self,
        properties: Dict[str, Any],
        parameters: Dict[str, Any],
        mask_key: str = "pred_masks",
        score_key: str = "pred_mask_scores",
        label_key: str = "pred_mask_labels",
        data_key: str = "data",
        box_key: Optional[str] = "pred_boxes",
        device: Optional[Union[torch.device, str]] = None,
        **kwargs,
    ):
        """
        Ensemble bounding box detections from tta and multiple models

        Args:
            properties: properties of the patient/case (e.g. tranpose axes)
            parameters: parameters for ensembling
            mask_key: key where masks are located inside prediction dict
            score_key: key where scores are located inside prediction dict
            label_key: key where labels are located inside prediction dict
            data_key: key where data is located inside batch dict
            box_key: key where boxes are located inside prediction dict
            device: device to use for internal computations
            kwargs: passed to super class
        """
        super().__init__(
            properties=properties,
            parameters=parameters,
            device=device,
            data_key=data_key,
            mask_key=mask_key,
            score_key=score_key,
            label_key=label_key,
            box_key=box_key,
            **kwargs,
        )
        self.overlap_map = None
        self.plateau_length = 0.5

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

                `interpolation_mode`: interpolation to convert between roi
                    masks into image
                `align_corners`: align corners during interpolation of roi
                    masks into image
                `bin_mask_threshold`: threshold applied after resizing of
                    roi mask to binary mask
        """
        return {
            # single model
            "model_iou": 0.1,
            "model_nms_fn": batched_weighted_nms,
            "model_score_thresh": 0.0,
            "model_topk": 1000,
            "model_detections_per_image": 100,
            # ensemble multiple models
            "ensemble_iou": 0.5,
            "ensemble_nms_fn": batched_nms,
            "ensemble_topk": 1000,
            "ensemble_num_preds": 30,  # FIXME
            "remove_small_boxes": 1e-2,
            "ensemble_score_thresh": 0.0,
            "interpolation_mode": "linear",
            "align_corners": None,
            "bin_mask_threshold": 0.5,
        }

    @classmethod
    def sweep_parameters(cls) -> Tuple[Dict[str, Any], Dict[str, Sequence[Any]]]:
        # iou_threshs = np.linspace(0.0, 0.8, 9)
        iou_threshs = np.linspace(0.0, 0.5, 6)
        iou_threshs[0] = 1e-5
        small_boxes_thresh = [1e-2] + np.linspace(2.0, 7.0, 6).tolist()

        param_sweep = {
            # single model
            "model_iou": iou_threshs,
            "model_nms_fn": [
                batched_nms,
                batched_weighted_nms,
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
        masks = [r.float().cpu() for r in result[self.mask_key]]
        scores = [r.float().cpu() for r in result[self.score_key]]
        labels = [r.float().cpu() for r in result[self.label_key]]
        boxes = [r.float().cpu() for r in result[self.box_key]]
        centers = [
            ops_torch.box_center(img_boxes) if img_boxes.numel() > 0 else torch.Tensor([]).to(img_boxes)
            for img_boxes in boxes
        ]
        tile_origins = [to for to in zip(*batch["tile_origin"])]

        tile_size = batch[self.data_key].shape[2:]
        weights = [self._get_box_in_tile_weight(c, tile_size) for c in centers]
        weights = [w * self.model_weights[self.model_current] for w in weights]
        boxes = apply_offsets_to_boxes(boxes, tile_origins)

        self.model_results[self.model_current]["masks"].extend(masks)
        self.model_results[self.model_current]["boxes"].extend(boxes)
        self.model_results[self.model_current]["scores"].extend(scores)
        self.model_results[self.model_current]["labels"].extend(labels)
        self.model_results[self.model_current]["weights"].extend(weights)

    def _get_box_in_tile_weight(
        self,
        box_centers: torch.Tensor,
        tile_size: Sequence[int],
    ) -> torch.Tensor:
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
        return get_box_in_tile_weight_linear(
            box_centers=box_centers,
            tile_size=tile_size,
            plateau_length=self.plateau_length,
        )

    def process_model(
        self, name: Hashable
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Í
        Process the output of a single model on the whole scan
        topk candidates -> nms

        Args:
            name: name of model to process

        Returns:
            Tensor: processed boxes
            Tensor: processed masks
            Tensor: processed probs
            Tensor: processed labels
            Tensor: processed weights
        """
        # collect predictions on whole case and apply postprocessing
        masks = cat(self.model_results[name]["masks"]).to(self.device)
        boxes = cat(self.model_results[name]["boxes"]).to(self.device)
        probs = cat(self.model_results[name]["scores"]).to(self.device)
        labels = cat(self.model_results[name]["labels"]).to(self.device)
        weights = cat(self.model_results[name]["weights"]).to(self.device)

        return self.postprocess_image(
            boxes=boxes,
            masks=masks,
            probs=probs,
            labels=labels,
            weights=weights,
            shape=tuple(self.properties["shape"]),
        )

    def postprocess_image(
        self,
        boxes: torch.Tensor,
        masks: torch.Tensor,
        probs: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
        shape: Optional[Tuple[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Postprocessing of a single image
        select topk predictions -> score threshold -> clipping -> \
            remove small boxes -> nms

        Args:
            boxes: predicted boxes [N, dim * 2]
            masks: predicted masks [N, dims]
            probs: predicted logits for boxes [N]
            labels: predicted labels for boxes [N]
            weights: weight for each box [N]

        Returns:
            Tensor: postprocessed boxes
            Tensor: postprocessed masks
            Tensor: postprocessed probs
            Tensor: postprocessed labels
            Tensor: postprocessed weights
        """
        assert masks.ndim == (boxes.shape[1] // 2) + 1, f"Found mask with {masks.ndim} and boxes with {boxes.shape[1]}"
        p_sorted, idx_sorted = probs.sort(descending=True)
        idx_sorted = idx_sorted[: self.parameters["model_topk"]]
        p_sorted = p_sorted[: self.parameters["model_topk"]]
        keep_idxs = p_sorted > self.parameters["model_score_thresh"]
        idx_sorted = idx_sorted[keep_idxs]

        b, m, p, l, w = (
            boxes[idx_sorted],
            masks[idx_sorted],
            probs[idx_sorted],
            labels[idx_sorted],
            weights[idx_sorted],
        )

        # b = clip_boxes_to_image(b, shape)
        # After clipping we could have boxes with volume 0 which we definitely
        # need to remove because of the IoU computation
        keep = ops_torch.remove_small_boxes(b, min_size=self.parameters["remove_small_boxes"])
        b, m, p, l, w = b[keep], m[keep], p[keep], l[keep], w[keep]

        _boxes, _masks, _probs, _labels, _weights = self.parameters["model_nms_fn"](
            boxes=b,
            scores=p,
            labels=l,
            weights=w,
            iou_thresh=self.parameters["model_iou"],
            masks=m,
        )

        # predictions are sorted
        _boxes = _boxes[: self.parameters.get("model_detections_per_image", 1000)]
        _masks = _masks[: self.parameters.get("model_detections_per_image", 1000)]
        _probs = _probs[: self.parameters.get("model_detections_per_image", 1000)]
        _labels = _labels[: self.parameters.get("model_detections_per_image", 1000)]
        _weights = _weights[: self.parameters.get("model_detections_per_image", 1000)]
        return _boxes, _masks, _probs, _labels, _weights

    def process_ensemble(
        self,
        boxes: List[torch.Tensor],
        masks: List[torch.Tensor],
        probs: List[torch.Tensor],
        labels: List[torch.Tensor],
        weights: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ensemble predictions from multiple models

        Args:
            boxes: predicted boxes List[[N, dims * 2]]
                (x1, y1, x2, y2, (z1, z2))
            masks: predicted masks List[N, dims]
            probs: predicted probabilities List[[N]]
            labels: predicted label List[[N]]
            weights: additional weight List[[N]]

        Returns:
            Tensor: ensembled box predictions
            Tensor: ensembled mask predictions
            Tensor: ensembled probabilities
            Tensor: ensembled labels
        """
        # num_models = len(boxes)
        boxes = cat(boxes, dim=0)
        masks = cat(masks, dim=0)
        probs = cat(probs, dim=0)
        labels = cat(labels, dim=0)
        weights = cat(weights, dim=0)

        assert masks.ndim == (boxes.shape[1] // 2) + 1, f"Found mask with {masks.ndim} and boxes with {boxes.shape[1]}"

        _, idx = probs.sort(descending=True)
        idx = idx[: self.parameters["ensemble_topk"]]
        boxes = boxes[idx]
        probs = probs[idx]
        labels = labels[idx]
        weights = weights[idx]

        # n_exp_preds = torch.tensor([num_models] * len(boxes)).to(boxes)
        # if "wbc" in self.parameters["ensemble_nms_fn"].__name__:
        #     _kwargs = {"n_exp_preds": n_exp_preds}
        # else:
        #     _kwargs = {}

        boxes, masks, probs, labels, _ = self.parameters["ensemble_nms_fn"](
            boxes,
            probs,
            labels,
            weights=weights,
            iou_thresh=self.parameters["ensemble_iou"],
            masks=masks,
            # **_kwargs,
        )

        keep = probs > self.parameters["ensemble_score_thresh"]
        boxes = boxes[keep]
        probs = probs[keep]
        labels = labels[keep]
        masks = masks[keep]

        num_topk = min(self.parameters["ensemble_num_preds"], boxes.size(0))
        _, idx = probs.sort(descending=True)
        keep_idx = idx[:num_topk]

        boxes = boxes[keep_idx]
        probs = probs[keep_idx]
        labels = labels[keep_idx]
        masks = masks[keep_idx]

        return boxes.cpu(), masks.cpu(), probs.cpu(), labels.cpu()

    def save_state(
        self,
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
            masks = cat(self.model_results[model]["masks"])
            probs = cat(self.model_results[model]["scores"])
            labels = cat(self.model_results[model]["labels"])
            weights = cat(self.model_results[model]["weights"])

            if len(probs) > self.parameters["model_topk"]:
                _, idx_sorted = probs.sort(descending=True)
                idx_sorted = idx_sorted[: self.parameters["model_topk"]]
                self.model_results[model]["boxes"] = boxes[idx_sorted]
                self.model_results[model]["masks"] = masks[idx_sorted]
                self.model_results[model]["scores"] = probs[idx_sorted]
                self.model_results[model]["labels"] = labels[idx_sorted]
                self.model_results[model]["weights"] = weights[idx_sorted]
        return super().save_state(target_dir=target_dir, name=name, **kwargs)
