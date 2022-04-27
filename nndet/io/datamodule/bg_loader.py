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

import os
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from loguru import logger
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase

from nndet.io.datamodule import DATALOADER_REGISTRY
from nndet.io.load import load_pickle
from nndet.io.patching import save_get_crop
from nndet.utils.info import maybe_verbose_iterable
from nndet.core.boxes.ops_np import box_size_np


class FixedSlimDataLoaderBase(SlimDataLoaderBase):
    def __init__(self,
                 *args,
                 num_batches_per_epoch: int = 2500,
                 **kwargs,
                 ):
        self.num_batches_per_epoch = num_batches_per_epoch
        super().__init__(*args, **kwargs)
    
    def __len__(self):
        return self.num_batches_per_epoch


@DATALOADER_REGISTRY.register
class DataLoader3DFast(FixedSlimDataLoaderBase):
    def __init__(self,
                 data: Dict,
                 batch_size: int,
                 patch_size_generator: Sequence[int],
                 patch_size_final: Sequence[int],
                 oversample_foreground_percent: float = 0.5,
                 memmap_mode: str = "r+",
                 pad_mode: str = "constant",
                 pad_kwargs_data: Optional[Dict[str, Any]] = None,
                 num_batches_per_epoch: int = 2500,
                 ):
        """
        Basic Dataloder for 3D Data.
        Center of foreground patches is sampled from pre computed bounding
        boxes. Background patches are sampled randomly. Cases are selected
        randomly.

        Args:
            data: dict with cases and data paths
            batch_size: size of batches to generate
            patch_size_generator: patch size prduced by the dataloader
            patch_size_final: final patch size after spatial transform
            oversample_foreground_percent: Oversample foreground patches.
                Each batch will be balanced to fullfill this criterion.
            memmap_mode: Do not change this. Defaults to "r".
            pad_mode: Padding mode for data. Defaults to "constant".
            pad_kwargs_data: Addition kwargs for data padding. Defaults to None.

        Raises:
            ValueError: patch size of dataloder and final patch size need to
                have the same length
        """
        super().__init__(
            data=data,
            batch_size=batch_size,
            number_of_threads_in_multithreaded=None,
            num_batches_per_epoch=num_batches_per_epoch,
        )
        if len(patch_size_generator) != len(patch_size_final):
            raise ValueError(f"Final and generator patch size need to have the same length."
                             f"Found generator {patch_size_generator} and "
                             f"final {patch_size_final} patch size.")
        self.patch_size_generator = patch_size_generator
        self.patch_size_final = patch_size_final
        self.oversample_foreground_percent = oversample_foreground_percent

        self.memmap_mode = memmap_mode

        self.pad_mode = pad_mode
        self.pad_kwargs_data = pad_kwargs_data if pad_kwargs_data is not None else {}

        # we sample bigger patches and create a center crop during augmentation
        # to cover the boarders of the patient we need to adjust the position
        self.need_to_pad = (np.array(patch_size_generator) - np.array(patch_size_final)).astype(np.int32)
        self.data_shape_batch, self.seg_shape_batch = self.determine_shapes()
        self.cache = self.build_cache()
        self.candidates_key = "boxes_file"

    def determine_shapes(self) -> Tuple[Tuple[int], Tuple[int]]:
        """
        Determines data and segmentation shape to preallocate arrays
        during loading

        Raises:
            RuntimeError: Raised if data was not unpacked

        Returns:
            Tuple[Tuple[int], Tuple[int]]: Final shape of data,
                Final shape of seg (including batchdim)
        """
        k = list(self._data.keys())[0]
        if (p := Path(self._data[k]['data_file'])).is_file():
            data = np.load(str(p), self.memmap_mode, allow_pickle=False)
        else:
            raise RuntimeError("You shall not pass! Unpack data first!")

        if (p := Path(self._data[k]['seg_file'])).is_file():
            seg = np.load(str(p), self.memmap_mode, allow_pickle=False)
        else:
            raise RuntimeError("You shall not pass! Unpack data first!")

        num_data_channels = data.shape[0]
        num_seg_channels = seg.shape[0]
        data_shape = (self.batch_size, num_data_channels, *self.patch_size_generator)
        seg_shape = (self.batch_size, num_seg_channels, *self.patch_size_generator)
        return data_shape, seg_shape

    def build_cache(self) -> Dict[str, List]:
        """
        Build up cache for sampling

        Returns:
            Dict[str, List]: cache for sampling
                `case`: list with all case identifiers
                `instances`: list with tuple of (case_id, instance_id)
        """
        instance_cache = []

        logger.info("Building Sampling Cache for Dataloder")
        for case_id, item in maybe_verbose_iterable(self._data.items(), desc="Sampling Cache"):
            instances = load_pickle(item['boxes_file'])["instances"]
            if instances:
                for instance_id in instances:
                    instance_cache.append((case_id, instance_id))
        return {"case": list(self._data.keys()), "instances": instance_cache}

    def select(self) -> Tuple[List, List]:
        """
        Selects cases and instances. If instance id is -1 a random background
        patch will be sampled.

        Foreground sampling: sample uniformly from all the foreground classes
            and enforce the respective class while patch sampling.
        Background sampling: We jsut sample a random case
        
        Returns:
            List: case identifiers
            List: instance ids
                id > 0 indicates an instance
                id = -1 indicates a random (background) patch
        """
        selected_cases = []
        selected_instances = []

        for idx in range(self.batch_size):
            if idx < round(self.batch_size * (1 - self.oversample_foreground_percent)):
                # sample bg / random case
                selected_cases.append(np.random.choice(self.cache["case"]))
                selected_instances.append(-1)
            else:
                # sample fg / select an instance
                idx = np.random.choice(range(len(self.cache["instances"])))
                _case, _instance_id = self.cache["instances"][idx]
                selected_cases.append(_case)
                selected_instances.append(int(_instance_id))
        return selected_cases, selected_instances

    def generate_train_batch(self) -> Dict[str, Any]:
        """
        Generate a single batch

        Returns:
            Dict: batch dict
                `data` (np.ndarray): data
                `seg` (np.ndarray): unordered(!) instance segmentation
                    Reordering needs to happen after final crop
                `instances` (List[Sequence[int]]): class for each instance in
                    the case (<- we can not extract them because we do not
                    know the present instances yet)
                `properties`(List[Dict]): properties of each case
                `keys` (List[str]): case ids
        """
        data_batch = np.zeros(self.data_shape_batch, dtype=float)
        seg_batch = np.zeros(self.seg_shape_batch, dtype=float)
        instances_batch, properties_batch, case_ids_batch = [], [], []

        selected_cases, selected_instances = self.select()
        for batch_idx, (case_id, instance_id) in enumerate(zip(selected_cases, selected_instances)):
            # print(case_id, instance_id)
            case_data = np.load(self._data[case_id]['data_file'], self.memmap_mode, allow_pickle=True)
            case_seg = np.load(self._data[case_id]['seg_file'], self.memmap_mode, allow_pickle=True)
            properties = load_pickle(self._data[case_id]['properties_file'])

            if instance_id < 0:
                candidates = self.load_candidates(case_id=case_id, fg_crop=False)
                crop = self.get_bg_crop(
                    case_data=case_data,
                    case_seg=case_seg,
                    properties=properties,
                    case_id=case_id,
                    candidates=candidates,
                )
            else:
                candidates = self.load_candidates(case_id=case_id, fg_crop=True)
                crop = self.get_fg_crop(
                    case_data=case_data,
                    case_seg=case_seg,
                    properties=properties,
                    case_id=case_id,
                    instance_id=instance_id,
                    candidates=candidates,
                )

            data_batch[batch_idx] = save_get_crop(case_data,
                                                  crop=crop,
                                                  mode=self.pad_mode,
                                                  **self.pad_kwargs_data,
                                                  )[0]
            seg_batch[batch_idx] = save_get_crop(case_seg,
                                                 crop=crop,
                                                 mode='constant',
                                                 constant_values=-1,
                                                 )[0]
            case_ids_batch.append(case_id)
            instances_batch.append(properties.pop("instances"))
            properties_batch.append(properties)

        return {'data': data_batch,
                'seg': seg_batch,
                'properties': properties_batch,
                'instance_mapping': instances_batch,
                'keys': case_ids_batch,
                }

    def load_candidates(self, case_id: str, fg_crop: bool) -> Union[Dict, None]:
        """
        Load candidates for sampling

        Args:
            case_id: case id to load candidates from
            fg_crop: True if foreground crop will be sampled, False if
                background will be sampled

        Returns:
            Union[Dict, None]: dict if fg, None if bg
        """
        if fg_crop:
            return load_pickle(self._data[case_id]['boxes_file'])
        else:
            return None

    def get_fg_crop(self,
                    case_data: np.ndarray,
                    case_seg: np.ndarray,
                    properties: dict,
                    case_id: str,
                    instance_id: int,
                    candidates: Union[Dict, None],
                    ) -> List[slice]:
        """
        Sample foreground patches from precomputed boxes

        Args:
            case_data: case data (this should be a memmap!)
            case_seg: case segmentation (this should be a memmap!)
            properties: properties of case
            case_id: identifier of case
            instance_id: instance index to sample
            candidates: candidate positions to sample foreground from.
                Should not be None for this case.

        Returns:
            List[slice]: determined crop
        """
        assert candidates is not None
        # some instances might get lost during resampling so we need to find the correct index
        idx = candidates["instances"].index(instance_id)
        box = candidates["boxes"][idx]  # [6]
        origin0 = np.random.randint(int(box[0]) + 1, int(box[2])) - (self.patch_size_generator[0] // 2)
        origin1 = np.random.randint(int(box[1]) + 1, int(box[3])) - (self.patch_size_generator[1] // 2)
        origin2 = np.random.randint(int(box[4]) + 1, int(box[5])) - (self.patch_size_generator[2] // 2)
        return [slice(origin0, origin0 + self.patch_size_generator[0]),
                slice(origin1, origin1 + self.patch_size_generator[1]),
                slice(origin2, origin2 + self.patch_size_generator[2])]

    def get_bg_crop(self,
                    case_data: np.ndarray,
                    case_seg: np.ndarray,
                    properties: dict,
                    case_id: str,
                    candidates: Union[Dict, None],
                    ) -> List[slice]:
        """
        Extract slices for (random) background crop

        Args:
            case_data: case data (this should be a memmap!)
            case_seg: case segmentation (this should be a memmap!)
            properties: properties of case
            case_id: identifier of case
            candidates: foreground candidates. Is not used in this
                specific implementation and thus None

        Returns:
            List[slice]: determined crop
        """
        data_shape = case_data.shape[1:]

        crop = []
        for ps, ds, _pad in zip(self.patch_size_generator, data_shape, self.need_to_pad):
            pad = _pad
            if pad + ds < ps:
                pad = ps - ds
            origin = np.random.randint(-(pad // 2), ds + (pad // 2) + (pad % 2) - ps + 1)
            crop.append(slice(origin, origin + ps))
        return crop


@DATALOADER_REGISTRY.register
class DataLoader3DOffset(DataLoader3DFast):
    def get_fg_crop(self,
                    case_data: np.ndarray,
                    case_seg: np.ndarray,
                    properties: dict,
                    case_id: str,
                    instance_id: int,
                    candidates: Union[Dict, None],
                    ) -> List[slice]:
        """
        Sample foreground patches from precomputed boxes

        Args:
            case_data: case data (this should be a memmap!)
            case_seg: case segmentation (this should be a memmap!)
            properties: properties of case
            case_id: identifier of case
            instance_id: instance index to sample
            candidates: candidate positions to sample foreground from.
                Should not be None for this case.

        Returns:
            List[slice]: determined crop
        """
        spatial_shape = case_data.shape[1:]
        # some instances might get lost during resampling so we need to find the correct index
        idx = candidates["instances"].index(instance_id)
        box = candidates["boxes"][[idx]]  # [1, 6]
        box_size = box_size_np(box)[0]
        box = box[0]

        origins = []
        for i, (ib, ib2) in enumerate([(0, 2), (1, 3), (4, 5)]):
            if spatial_shape[i] <= self.patch_size_generator[i]:  # patch larger than scan
                # we center the slice and pad the rest
                origins.append(- (self.need_to_pad[i] // 2))
            elif box_size[i] >= self.patch_size_final[i]:  # selected instance is larger than patch
                # we can not offset, we select our center point inside the bounding box and hope for the best
                center = np.random.randint(int(box[ib]) + 1, int(box[ib2]))
                origins.append(center - (self.patch_size_generator[i] // 2))
            else:  # create best effort offset
                patch_upper_bound = spatial_shape[i] - self.patch_size_final[i]
                lower_bound = np.clip(box[ib] - (self.patch_size_final[i] - box_size[i]),
                                      a_min=0, a_max=patch_upper_bound)
                upper_bound = np.clip(box[ib], a_min=0, a_max=patch_upper_bound)

                if lower_bound == upper_bound:
                    _origin = int(lower_bound)
                else:
                    _origin = np.random.randint(lower_bound, upper_bound)

                origins.append(_origin - (self.need_to_pad[i] // 2))

        return [slice(origins[0], origins[0] + self.patch_size_generator[0]),
                slice(origins[1], origins[1] + self.patch_size_generator[1]),
                slice(origins[2], origins[2] + self.patch_size_generator[2]),
                ]


@DATALOADER_REGISTRY.register
class DataLoader3DBalanced(DataLoader3DOffset):
    def build_cache(self) -> Tuple[Dict[int, List[Tuple[str, int]]], List]:
        """
        Build up cache for sampling

        Returns:
            Dict[int, List[Tuple[str, int]]]: foreground cache which contains
                of list of tuple of case ids and instance ids for each class
            List: background cache (all samples which do not have any
                foreground)
        """
        fg_cache = defaultdict(list)

        logger.info("Building Sampling Cache for Dataloder")
        for case_id, item in maybe_verbose_iterable(self._data.items(), desc="Sampling Cache"):
            candidates = load_pickle(item['boxes_file'])
            if candidates["instances"]:
                for instance_id, instance_class in zip(candidates["instances"], candidates["labels"]):
                    fg_cache[int(instance_class)].append((case_id, instance_id))
        return {"fg": fg_cache, "case": list(self._data.keys())}

    def select(self) -> Tuple[List, List]:
        """
        Foreground sampling: sample uniformly from all the foreground classes
            and enforce the respective class while patch sampling.
        Background sampling: We jsut sample a random case
        """
        selected_classes = np.random.choice(
            list(self.cache["fg"].keys()), self.batch_size, replace=True)

        selected_cases = []
        selected_instances = []
        for idx in range(len(selected_classes)):
            if idx < round(self.batch_size * (1 - self.oversample_foreground_percent)):
                # sample bg / random case
                selected_cases.append(np.random.choice(self.cache["case"]))
                selected_instances.append(-1)
            else:
                # sample fg / select an instance
                _i = np.random.choice(range(len(self.cache["fg"][selected_classes[idx]])))
                _case, _instance_id = self.cache["fg"][selected_classes[idx]][_i]
                selected_cases.append(_case)
                selected_instances.append(int(_instance_id))
        return selected_cases, selected_instances
