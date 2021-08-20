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
from typing import Dict, Optional, Tuple, Sequence, Any

import torch
import numpy as np
from loguru import logger
from scipy.ndimage import gaussian_filter
from torch import Tensor

from nndet.inference.ensembler.base import BaseEnsembler
from nndet.inference.restore import restore_fmap
from nndet.utils.info import experimental


class SegmentationEnsembler(BaseEnsembler):
    ID = "seg"

    @experimental
    def __init__(self,
                 seg_key: str = 'pred_seg',
                 data_key: str = 'data',
                 **kwargs,
                 ):
        """
        Ensemble segmentation predictions from tta and model ensembling

        Args:
            seg_key: key where segmentation is located inside prediction dict
            data_key: key where data is located inside batch dict
            use_gaussian: apply gaussian weighting to individual crops
            non_lin: non linearity to apply to convert logits to probabilities
                (will be applied after consolidation)
            argmax: apply argmax to output
            kwargs: passed to super class
        """
        super().__init__(**kwargs)
        self.seg_key = seg_key
        self.data_key = data_key

        self.model_results: Optional[Tensor] = None
        self.overlap = torch.zeros(self.properties["shape"])
        self.cache_crop_weight: Dict[Tuple, torch.Tensor] = {}

    @classmethod
    def from_case(cls,
                  case: Dict,
                  properties: Dict,
                  parameters: Optional[Dict] = None,
                  seg_key: str = 'pred_seg',
                  data_key: str = 'data',
                  **kwargs,
                  ):
        """
        Primary way to instantiate this class. Automatically extracts all
        properties and uses a default set of parameters for ensembling.

        Args:
            case: case which is predicted.
            mode: operation mode of ensembler (defines which network was used)
                e.g. '2d' | '3d'
            properties: Additional properties.
                Required keys:
                    `transpose_backward`
                    `spacing_after_resampling`
                    `crop_bbox`
            parameters: Additional parameters. Defaults to None.
            seg_key: key where segmentation is located inside prediction dict
            data_key: key where data is located inside batch dict
        """
        parameters = parameters if parameters is not None else {}
        _parameters = {"use_gaussian": True, "argmax": True}
        _parameters.update(parameters)

        _properties = {
            "shape": case[data_key].shape[1:],  # remove channel dim
            "transpose_backward": properties["transpose_backward"],
            "original_spacing": properties["original_spacing"],
            "spacing_after_resampling": properties["spacing_after_resampling"],
            "crop_bbox": properties["crop_bbox"],
            "size_after_cropping": properties["size_after_cropping"],
            "original_size_before_cropping": properties["original_size_of_raw_data"],
            "itk_origin": properties["itk_origin"],
            "itk_spacing": properties["itk_spacing"],
            "itk_direction": properties["itk_direction"],
        }

        return cls(
            properties=_properties,
            parameters=_parameters,
            seg_key=seg_key,
            data_key=data_key,
            **kwargs,
            )

    def add_model(self,
                  name: Optional[str] = None,
                  model_weight: Optional[float] = None,
                  ) -> str:
        """
        This functions signales the ensembler to add a new model for internal
        processing

        Args:
            name: Name of the model. If None, uses counts the models.
            model_weight: Optional weight for this model. Defaults to None.
        """
        if name is None:
            name = len(self.model_weights) + 1
        if name in self.model_weights:
            raise ValueError(f"Invalid model name, model {name} is already present")

        if model_weight is None:
            model_weight = 1.0

        self.model_weights[name] = model_weight
        self.model_current = name
        return name

    @classmethod
    def sweep_parameters(cls) -> Tuple[Dict[str, Any], Dict[str, Sequence[Any]]]:
        """
        Not available for segmentation ensembler

        Returns:
            Dict[str, Sequence[Any]]: Parameters to sweep. The keys define the
                parameters wile the Sequences are the values to sweep.
        """
        return {}, {}

    @torch.no_grad()
    def process_batch(self, result: Dict, batch: Dict):
        """
        Process a single batch of bounding box predictions

        Args:
            result: prediction from detector. Need to provide boxes, scores
                and class labels
                    `self.seg_key`: [Tensor]: predicted segmentation [N, C, dims]
            batch: input batch
                `tile_origin: origin of crop with recard to actual data (
                    in case of padding)
                `crop`: Sequence[slice] original crop from data
        """
        seg_batch = result[self.seg_key].cpu()
        crops = batch["crop"]

        weight = self.get_weighting(tuple(seg_batch.shape[2:])).to(seg_batch)
        seg_batch = seg_batch * weight[None].to(seg_batch) * self.model_weights[self.model_current]

        if self.model_results is None:
            self.model_results = torch.zeros(
                (int(seg_batch.shape[1]), *self.properties["shape"])).to(seg_batch)

        for seg, crop in zip(seg_batch, zip(*crops)):
            _weight = weight.clone()
            seg, case_crop = self.crop_to_case_boundaries(seg, crop)
            _weight, case_crop2 = self.crop_to_case_boundaries(_weight[None], crop)
            assert case_crop == case_crop2
            self.model_results[case_crop] += seg
            self.overlap[case_crop] += _weight[0]

    def crop_to_case_boundaries(self, seg: torch.Tensor, crop: Sequence[slice]):
        """
        In case padding was used at the borders, the padding needs to be removed

        Args
            seg: predicted segmentation
            Sequence[slice]: crop in case to save segmentation
        """
        if len(crop) > self.model_results.ndim - 1:
            crop = crop[-(self.model_results.ndim - 1):]

        crop_slicer = []
        case_slicer = []
        for dim, c in enumerate(crop):
            case_start = max(0, c.start)
            case_stop = min(self.model_results.shape[dim + 1], c.stop)

            diff_stop = c.stop - self.model_results.shape[dim + 1]
            crop_start = max(0, 0 - (c.start - 0))  # 0 added for completeness of pattern
            crop_stop = min(seg.shape[dim + 1], seg.shape[dim + 1] - diff_stop)

            crop_slicer.append(slice(crop_start, crop_stop, c.step))
            case_slicer.append(slice(case_start, case_stop, c.step))
        return seg[(..., *crop_slicer)], (..., *case_slicer)

    def get_weighting(self, crop_size: Tuple[int]) -> torch.Tensor:
        """
        Get matrix to weight predictions inside a single patch

        Args:
            crop_size: size of crop

        Returns:
            Tensor: weight for crop
            Tuple[int]: size of patch
        """
        if crop_size not in self.cache_crop_weight:
            if self.parameters["use_gaussian"]:
                logger.info(f"Creating new gaussian weight matrix for crop size {crop_size}")
                tmp = np.zeros(crop_size)
                center_coords = [i // 2 for i in crop_size]
                sigmas = [i // 8 for i in crop_size]
                tmp[tuple(center_coords)] = 1
                tmp_smooth = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
                tmp_smooth = tmp_smooth / tmp_smooth.max() * 1
                weighting = tmp_smooth + 1e-8
                self.cache_crop_weight[crop_size] = torch.from_numpy(weighting).float()
            else:
                logger.info(f"Creating new weight matrix for crop size {crop_size}")
                self.cache_crop_weight[crop_size] = torch.ones(crop_size, dtype=torch.float)

        return self.cache_crop_weight[crop_size]

    def restore_prediction(self, logit_maps: Tensor) -> Tensor:
        """
        Restore predictions in the original image space

        Args:
            boxes: predicted boxes [N, dims * 2] (x1, y1, x2, y2, (z1, z2))

        Returns:
            Tensor: boxes in original image space [N, dims * 2]
                (x1, y1, x2, y2, (z1, z2))
        """
        _old_dtype = logit_maps.dtype
        logit_maps_np = restore_fmap(
                fmap=logit_maps.detach().cpu().numpy(),
                transpose_backward=self.properties["transpose_backward"],
                original_spacing=self.properties["original_spacing"],
                spacing_after_resampling=self.properties["spacing_after_resampling"],
                original_size_before_cropping=self.properties["original_size_before_cropping"],
                size_after_cropping=self.properties["size_after_cropping"],
                crop_bbox=self.properties["crop_bbox"],
                interpolation_order=1,
                interpolation_order_z=0,
                do_separate_z=None,
        )
        logit_maps = torch.from_numpy(logit_maps_np).to(dtype=_old_dtype)
        return logit_maps

    @torch.no_grad()
    def get_case_result(self,
                        restore: bool = False, **kwargs
                        ) -> Dict[str, Tensor]:
        """
        Get final result for case after ensembling and TTA

        Returns:
            Dict: results
                `pred_seg`: [C, dims] if :param:`self.argmax`
                    if False and [dims] if True
                `restore`: indicate whether predictions were restored in
                    original image space
                `itk_origin`: itk origin of image before preprocessing
                `itk_spacing`: itk spacing of image before preprocessing
                `itk_direction`: itk direction of image before preprocessing
        """
        result = self.model_results / self.overlap[None]

        if restore:
            result = self.restore_prediction(result)

        if self.parameters["argmax"]:
            result = result.argmax(dim=0).to(dtype=torch.uint8)

        self.case_result = result
        return {
            "pred_seg": self.case_result,
            "restore": restore,
            "itk_origin": self.properties["itk_origin"],
            "itk_spacing": self.properties["itk_spacing"],
            "itk_direction": self.properties["itk_direction"],
            }

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
        """
        super().save_state(
            target_dir=target_dir,
            name=name,
            seg_key=self.seg_key,
            data_key=self.data_key,
            case_crop_weight=self.cache_crop_weight,
            **kwargs,
        )

    @classmethod
    def from_checkpoint(cls, base_dir: PathLike, case_id: str, **kwargs):
        ckp = torch.load(str(Path(base_dir) / f"{case_id}_{cls.ID}.pt"))
        t = cls(
            properties=ckp["properties"],
            parameters=ckp["parameters"],
            seg_key=ckp["seg_key"],
            data_key=ckp["data_key"],
        )
        t._load(ckp)
        return t
