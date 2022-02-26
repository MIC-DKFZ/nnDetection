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

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase

from nndet.io.datamodule import DATALOADER_REGISTRY
from nndet.io.datamodule.mixins.bgcrop import RandomBGCrop2D, RandomBGCrop3D
from nndet.io.datamodule.mixins.fgcrop import (
    InsideFGCrop3D,
    OffsetFGCrop2D,
    OffsetFGCrop3D,
)
from nndet.io.datamodule.mixins.select import (
    BalancedSelectionMixin,
    RandomSelectionMixin,
)
from nndet.io.load import load_pickle
from nndet.io.patching import save_get_crop
from nndet.utils.info import deprecate


class BaseDataLoader3D(SlimDataLoaderBase):
    build_cache: Callable[[], Dict]
    select: Callable[[], Tuple[List, List]]
    get_bg_crop: Callable[..., List[slice]]
    get_fg_crop: Callable[..., List[slice]]

    def __init__(
        self,
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
        )
        self.num_batches_per_epoch = num_batches_per_epoch
        if len(patch_size_generator) != len(patch_size_final):
            raise ValueError(
                f"Final and generator patch size need to have the same length."
                f"Found generator {patch_size_generator} and "
                f"final {patch_size_final} patch size."
            )
        self.patch_size_generator = patch_size_generator
        self.patch_size_final = patch_size_final
        self.oversample_foreground_percent = oversample_foreground_percent

        self.memmap_mode = memmap_mode

        self.pad_mode = pad_mode
        self.pad_kwargs_data = pad_kwargs_data if pad_kwargs_data is not None else {}

        # we sample bigger patches and create a center crop during augmentation
        # to cover the boarders of the patient we need to adjust the position
        self.need_to_pad = (
            np.array(patch_size_generator) - np.array(patch_size_final)
        ).astype(np.int32)
        self.data_shape_batch, self.seg_shape_batch = self.determine_shapes()
        self.cache = self.build_cache()
        self.candidates_key = "boxes_file"

    def __len__(self):
        return self.num_batches_per_epoch

    def determine_shapes(self) -> Tuple[Tuple[int], Tuple[int]]:
        """
        Determines data and segmentation shape to preallocate arrays
        during loading

        Raises:
            RuntimeError: Raised if data was not unpacked

        Returns:
            Tuple[int]: Final shape of data (including batchdim)
            Tuple[int]: Final shape of seg (including batchdim)
        """
        k = list(self._data.keys())[0]
        if (p := Path(self._data[k]["data_file"])).is_file():
            data = np.load(str(p), self.memmap_mode, allow_pickle=False)
        else:
            raise RuntimeError("You shall not pass! Unpack data first!")

        if (p := Path(self._data[k]["seg_file"])).is_file():
            seg = np.load(str(p), self.memmap_mode, allow_pickle=False)
        else:
            raise RuntimeError("You shall not pass! Unpack data first!")

        num_data_channels = data.shape[0]
        num_seg_channels = seg.shape[0]
        data_shape = (self.batch_size, num_data_channels, *self.patch_size_generator)
        seg_shape = (self.batch_size, num_seg_channels, *self.patch_size_generator)
        return data_shape, seg_shape

    def generate_train_batch(self) -> Dict[str, Any]:
        """
        Generate a single batch

        Returns:
            Dict: batch dict

                ``"data"`` np.ndarray
                    data

                ``"seg"`` np.ndarray
                    unordered(!) numbered instance segmentation
                    Reordering needs to happen after final crop

                ``"instances"`` List[Sequence[int]]
                    class for each instance in the case (<- we can not
                    extract them because we do not know the present instances
                    yet)

                ``"properties"`` List[Dict]
                    properties of each case

                ``"keys"`` List[str]
                    case ids

        """
        data_batch = np.zeros(self.data_shape_batch, dtype=float)
        seg_batch = np.zeros(self.seg_shape_batch, dtype=float)
        instances_batch, properties_batch, case_ids_batch = [], [], []

        selected_cases, selected_instances = self.select()
        for batch_idx, (case_id, instance_id) in enumerate(
            zip(selected_cases, selected_instances)
        ):
            # print(case_id, instance_id)
            case_data = np.load(
                self._data[case_id]["data_file"], self.memmap_mode, allow_pickle=True
            )
            case_seg = np.load(
                self._data[case_id]["seg_file"], self.memmap_mode, allow_pickle=True
            )
            properties = load_pickle(self._data[case_id]["properties_file"])

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

            data_batch[batch_idx] = save_get_crop(
                case_data,
                crop=crop,
                mode=self.pad_mode,
                **self.pad_kwargs_data,
            )[0]
            seg_batch[batch_idx] = save_get_crop(
                case_seg,
                crop=crop,
                mode="constant",
                constant_values=-1,
            )[0]
            case_ids_batch.append(case_id)
            instances_batch.append(properties.pop("instances"))
            properties_batch.append(properties)

        return {
            "data": data_batch,
            "seg": seg_batch,
            "properties": properties_batch,
            "instance_mapping": instances_batch,
            "keys": case_ids_batch,
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
            return load_pickle(self._data[case_id]["boxes_file"])
        else:
            return None


class BaseDataLoader2D(BaseDataLoader3D):
    def generate_train_batch(self) -> Dict[str, Any]:
        """
        Generate a single batch

        Returns:
            Dict: batch dict

                ``"data"`` np.ndarray
                    data

                ``"seg"`` np.ndarray
                    unordered(!) numbered instance segmentation
                    Reordering needs to happen after final crop

                ``"instances"`` List[Sequence[int]]
                    class for each instance in the case (<- we can not
                    extract them because we do not know the present instances
                    yet)

                ``"properties"`` List[Dict]
                    properties of each case

                ``"keys"`` List[str]
                    case ids
        """
        data_batch = np.zeros(self.data_shape_batch, dtype=np.float32)
        seg_batch = np.zeros(self.seg_shape_batch, dtype=np.float32)
        instances_batch, properties_batch, case_ids_batch = [], [], []

        selected_cases, selected_instances = self.select()
        for batch_idx, (case_id, instance_id) in enumerate(
            zip(selected_cases, selected_instances)
        ):
            case_data = np.load(
                self._data[case_id]["data_file"], self.memmap_mode, allow_pickle=False
            )
            case_seg = np.load(
                self._data[case_id]["seg_file"], self.memmap_mode, allow_pickle=False
            )
            properties = load_pickle(self._data[case_id]["properties_file"])
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

            data_batch[batch_idx] = save_get_crop(
                case_data,
                crop=crop,
                mode=self.pad_mode,
                **self.pad_kwargs_data,
            )[0][:, 0]
            seg_batch[batch_idx] = save_get_crop(
                case_seg,
                crop=crop,
                mode="constant",
                constant_values=-1,
            )[0][:, 0]
            case_ids_batch.append(case_id)
            instances_batch.append(properties.pop("instances"))
            properties_batch.append(properties)

        return {
            "data": data_batch,
            "seg": seg_batch,
            "properties": properties_batch,
            "instance_mapping": instances_batch,
            "keys": case_ids_batch,
        }


###
# Concrete Dataloader Classes
###
@DATALOADER_REGISTRY.register
class DataLoader3D(
    RandomBGCrop3D,
    InsideFGCrop3D,
    RandomSelectionMixin,
    BaseDataLoader3D,
):
    pass


@DATALOADER_REGISTRY.register
class DataLoader3DOffset(
    RandomBGCrop3D,
    OffsetFGCrop3D,
    RandomSelectionMixin,
    BaseDataLoader3D,
):
    pass


@DATALOADER_REGISTRY.register
class DataLoader3DOffsetBalanced(
    RandomBGCrop3D,
    OffsetFGCrop3D,
    BalancedSelectionMixin,
    BaseDataLoader3D,
):
    pass


@DATALOADER_REGISTRY.register
class DataLoader2DOffset(
    RandomBGCrop2D,
    OffsetFGCrop2D,
    RandomSelectionMixin,
    BaseDataLoader2D,
):
    pass


####
# Backwards Compatibility
####
@DATALOADER_REGISTRY.register
class DataLoader3DFast(DataLoader3D):  # backwards compatibility
    @deprecate(
        replacement="`DataLoader3D`",
        deprecate="v0.1.3",
        remove="v0.2",
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@DATALOADER_REGISTRY.register
class DataLoader3DPOB(DataLoader3DOffsetBalanced):
    @deprecate(
        replacement="`DataLoader3DOffset` with probability parameter",
        deprecate="v0.1.3",
        remove="v0.2",
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@DATALOADER_REGISTRY.register
class DataLoader3DBalanced(DataLoader3DOffsetBalanced):
    @deprecate(
        replacement="`DataLoader3DOffsetBalanced`",
        deprecate="v0.1.3",
        remove="v0.2",
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
