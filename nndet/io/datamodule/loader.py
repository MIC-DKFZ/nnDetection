# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase

from nndet.io.datamodule import DATALOADER_REGISTRY
from nndet.io.datamodule.mixins.bgcrop import RandomBGCrop2D, RandomBGCrop3D
from nndet.io.datamodule.mixins.fgcrop import (
    InsideFGCrop3D,
    OffsetFGCrop2D,
    OffsetFGCrop3D,
    OffsetFGCrop3DV2,
)
from nndet.io.datamodule.mixins.select import (
    ObjectBalancedSelectionMixin,
    PatientBalancedSelectionMixin,
    RandomSelectionMixin,
)
from nndet.io.load import load_pickle
from nndet.io.patching import save_get_crop
from nndet.utils.enums import SelectionMode
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
        num_batches_per_epoch: int = 2500,
    ):
        """
        Basic Dataloder for 3D Data.
        Center of foreground patches is sampled from pre computed bounding
        boxes.
        Background patches are sampled randomly.
        Objects are selected randomly.

        Args:
            data: dict with cases and data paths
            batch_size: size of batches to generate
            patch_size_generator: patch size prduced by the dataloader
            patch_size_final: final patch size after spatial transform
            oversample_foreground_percent: Oversample foreground patches.
                Each batch will be balanced to fullfill this criterion.
            memmap_mode: Do not change this. Defaults to "r".
            num_batches_per_epoch: number of batcher per epoch

        Raises:
            ValueError: patch size of dataloder and final patch size need to
                have the same length
        """
        super().__init__(
            data=data,
            batch_size=batch_size,
            number_of_threads_in_multithreaded=None,
        )
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
        self.num_batches_per_epoch = num_batches_per_epoch

        # we sample bigger patches and create a center crop during augmentation
        # to cover the boarders of the patient we need to adjust the position
        self.need_to_pad = (np.array(patch_size_generator) - np.array(patch_size_final)).astype(np.int32)
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
        for batch_idx, (case_id, instance_id) in enumerate(zip(selected_cases, selected_instances)):
            # print(case_id, instance_id)
            case_data = np.load(self._data[case_id]["data_file"], self.memmap_mode, allow_pickle=True)
            case_seg = np.load(self._data[case_id]["seg_file"], self.memmap_mode, allow_pickle=True)
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
                mode="constant",
                constant_values=0,
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
        for batch_idx, (case_id, instance_id) in enumerate(zip(selected_cases, selected_instances)):
            case_data = np.load(self._data[case_id]["data_file"], self.memmap_mode, allow_pickle=False)
            case_seg = np.load(self._data[case_id]["seg_file"], self.memmap_mode, allow_pickle=False)
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

            data_batch[batch_idx] = save_get_crop(case_data, crop=crop, mode="constant", constant_values=0,)[
                0
            ][:, 0]
            seg_batch[batch_idx] = save_get_crop(case_seg, crop=crop, mode="constant", constant_values=-1,)[
                0
            ][:, 0]
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
    def __init__(
        self,
        data: Dict,
        batch_size: int,
        patch_size_generator: Sequence[int],
        patch_size_final: Sequence[int],
        oversample_foreground_percent: float = 0.5,
        memmap_mode: str = "r+",
        num_batches_per_epoch: int = 2500,
        force_bg_case: bool = False,
        offset_prob: float = 1.0,
        offset_magn: float = 1.0,
    ):
        """
        Dataloder for 3D Data.
        Center of foreground patches are sampled within bounding boxes.
        Background patches are sampled randomly.
        Objects are selected randomly.

        Args:
            data: dict with cases and data paths
            batch_size: size of batches to generate
            patch_size_generator: patch size prduced by the dataloader
            patch_size_final: final patch size after spatial transform
            oversample_foreground_percent: Oversample foreground patches.
                Each batch will be balanced to fullfill this criterion.
            memmap_mode: Do not change this. Defaults to "r".
            num_batches_per_epoch: number of batcher per epoch
            force_bg_case: force extraction of background patches from cases
                without any objects
            offset_prob: probability to apply additional offsets of objects.
            offset_magn: magnitude of additional offset.

        Raises:
            ValueError: patch size of dataloder and final patch size need to
                have the same length

        Notes:
            Please refer to the Mixin-Classes for moe details about the
            patch extraction procedure.
        """
        super().__init__(
            data=data,
            batch_size=batch_size,
            patch_size_generator=patch_size_generator,
            patch_size_final=patch_size_final,
            oversample_foreground_percent=oversample_foreground_percent,
            memmap_mode=memmap_mode,
            num_batches_per_epoch=num_batches_per_epoch,
        )
        self.force_bg_case = force_bg_case
        self.offset_prob = offset_prob
        self.offset_magn = offset_magn


@DATALOADER_REGISTRY.register
class DataLoader3DOffset(
    RandomBGCrop3D,
    OffsetFGCrop3D,
    RandomSelectionMixin,
    BaseDataLoader3D,
):
    def __init__(
        self,
        data: Dict,
        batch_size: int,
        patch_size_generator: Sequence[int],
        patch_size_final: Sequence[int],
        oversample_foreground_percent: float = 0.5,
        memmap_mode: str = "r+",
        num_batches_per_epoch: int = 2500,
        force_bg_case: bool = False,
        offset_prob: float = 1.0,
        offset_magn: float = 1.0,
    ):
        """
        Dataloder for 3D Data.
        Center of foreground patches is sampled with an offset while objects
        reamin inside the patch.
        Background patches are sampled randomly.
        Objects are selected randomly.

        Args:
            data: dict with cases and data paths
            batch_size: size of batches to generate
            patch_size_generator: patch size prduced by the dataloader
            patch_size_final: final patch size after spatial transform
            oversample_foreground_percent: Oversample foreground patches.
                Each batch will be balanced to fullfill this criterion.
            memmap_mode: Do not change this. Defaults to "r".
            num_batches_per_epoch: number of batcher per epoch
            force_bg_case: force extraction of background patches from cases
                without any objects
            offset_prob: probability to apply additional offsets of objects.
            offset_magn: magnitude of additional offset.

        Raises:
            ValueError: patch size of dataloder and final patch size need to
                have the same length

        Notes:
            Please refer to the Mixin-Classes for moe details about the
            patch extraction procedure.
        """
        super().__init__(
            data=data,
            batch_size=batch_size,
            patch_size_generator=patch_size_generator,
            patch_size_final=patch_size_final,
            oversample_foreground_percent=oversample_foreground_percent,
            memmap_mode=memmap_mode,
            num_batches_per_epoch=num_batches_per_epoch,
        )
        self.force_bg_case = force_bg_case
        self.offset_prob = offset_prob
        self.offset_magn = offset_magn


@DATALOADER_REGISTRY.register
class DataLoader3DOffsetV2(
    RandomBGCrop3D,
    OffsetFGCrop3DV2,
    RandomSelectionMixin,
    BaseDataLoader3D,
):
    def __init__(
        self,
        data: Dict,
        batch_size: int,
        patch_size_generator: Sequence[int],
        patch_size_final: Sequence[int],
        oversample_foreground_percent: float = 0.5,
        memmap_mode: str = "r+",
        num_batches_per_epoch: int = 2500,
        force_bg_case: bool = False,
        offset_prob: float = 1.0,
        offset_magn: float = 1.0,
        max_size_pct: float = 1.0,
    ):
        """
        Dataloder for 3D Data.
        Center of foreground patches is sampled with an offset while objects
        reamin inside the patch.
        Background patches are sampled randomly.
        Objects are selected randomly.

        Args:
            data: dict with cases and data paths
            batch_size: size of batches to generate
            patch_size_generator: patch size prduced by the dataloader
            patch_size_final: final patch size after spatial transform
            oversample_foreground_percent: Oversample foreground patches.
                Each batch will be balanced to fullfill this criterion.
            memmap_mode: Do not change this. Defaults to "r".
            num_batches_per_epoch: number of batcher per epoch
            force_bg_case: force extraction of background patches from cases
                without any objects
            offset_prob: probability to apply additional offsets of objects.
            offset_magn: magnitude of additional offset.
            max_size_pct: if object size exceeds this percentage of the
                patch size the patch center will be sampled randomly
                within the box instead of an offeset.

        Raises:
            ValueError: patch size of dataloder and final patch size need to
                have the same length

        Notes:
            Please refer to the Mixin-Classes for moe details about the
            patch extraction procedure.
        """
        super().__init__(
            data=data,
            batch_size=batch_size,
            patch_size_generator=patch_size_generator,
            patch_size_final=patch_size_final,
            oversample_foreground_percent=oversample_foreground_percent,
            memmap_mode=memmap_mode,
            num_batches_per_epoch=num_batches_per_epoch,
        )
        self.force_bg_case = force_bg_case
        self.offset_prob = offset_prob
        self.offset_magn = offset_magn
        self.max_size_pct = max_size_pct


@DATALOADER_REGISTRY.register
class DataLoader3DOffsetObjectBalanced(
    RandomBGCrop3D,
    OffsetFGCrop3DV2,
    ObjectBalancedSelectionMixin,
    BaseDataLoader3D,
):
    def __init__(
        self,
        data: Dict,
        batch_size: int,
        patch_size_generator: Sequence[int],
        patch_size_final: Sequence[int],
        oversample_foreground_percent: float = 0.5,
        memmap_mode: str = "r+",
        num_batches_per_epoch: int = 2500,
        force_bg_case: bool = False,
        offset_prob: float = 1.0,
        offset_magn: float = 1.0,
        max_size_pct: float = 1.0,
        selection_mode: Union[str, SelectionMode] = "uniform",
    ):
        """
        Dataloder for 3D Data.
        Center of foreground patches is sampled with an offset while objects
        reamin inside the patch.
        Background patches are sampled randomly.
        Object classes are balanced and each object inside a class is sampled
        uniformly.

        Args:
            data: dict with cases and data paths
            batch_size: size of batches to generate
            patch_size_generator: patch size prduced by the dataloader
            patch_size_final: final patch size after spatial transform
            oversample_foreground_percent: Oversample foreground patches.
                Each batch will be balanced to fullfill this criterion.
            memmap_mode: Do not change this. Defaults to "r".
            num_batches_per_epoch: number of batcher per epoch
            force_bg_case: force extraction of background patches from cases
                without any objects
            offset_prob: probability to apply additional offsets of objects.
            offset_magn: magnitude of additional offset.
            max_size_pct: if object size exceeds this percentage of the
                patch size the patch center will be sampled randomly
                within the box instead of an offeset.
            selection_mode: Define how classes should be sampled. 'uniform'
                sampled each object class with the sample probability.
                'sqrt' applies sqrt to the number of objects per class and
                uses those values for weighted sampling.

        Raises:
            ValueError: patch size of dataloder and final patch size need to
                have the same length

        Notes:
            Please refer to the Mixin-Classes for moe details about the
            patch extraction procedure.
        """
        super().__init__(
            data=data,
            batch_size=batch_size,
            patch_size_generator=patch_size_generator,
            patch_size_final=patch_size_final,
            oversample_foreground_percent=oversample_foreground_percent,
            memmap_mode=memmap_mode,
            num_batches_per_epoch=num_batches_per_epoch,
        )
        self.force_bg_case = force_bg_case
        self.offset_prob = offset_prob
        self.offset_magn = offset_magn
        self.max_size_pct = max_size_pct
        self.selection_mode = selection_mode


@DATALOADER_REGISTRY.register
class DataLoader3DOffsetPatientBalanced(
    RandomBGCrop3D,
    OffsetFGCrop3DV2,
    PatientBalancedSelectionMixin,
    BaseDataLoader3D,
):
    def __init__(
        self,
        data: Dict,
        batch_size: int,
        patch_size_generator: Sequence[int],
        patch_size_final: Sequence[int],
        oversample_foreground_percent: float = 0.5,
        memmap_mode: str = "r+",
        num_batches_per_epoch: int = 2500,
        force_bg_case: bool = False,
        offset_prob: float = 1.0,
        offset_magn: float = 1.0,
        max_size_pct: float = 1.0,
        selection_mode: Union[str, SelectionMode] = "uniform",
    ):
        """
        Dataloder for 3D Data.
        Center of foreground patches is sampled with an offset while objects
        reamin inside the patch.
        Background patches are sampled randomly.
        Objects are selected randomly.

        Args:
            data: dict with cases and data paths
            batch_size: size of batches to generate
            patch_size_generator: patch size prduced by the dataloader
            patch_size_final: final patch size after spatial transform
            oversample_foreground_percent: Oversample foreground patches.
                Each batch will be balanced to fullfill this criterion.
            memmap_mode: Do not change this. Defaults to "r".
            num_batches_per_epoch: number of batcher per epoch
            force_bg_case: force extraction of background patches from cases
                without any objects
            offset_prob: probability to apply additional offsets of objects.
            offset_magn: magnitude of additional offset.
            max_size_pct: if object size exceeds this percentage of the
                patch size the patch center will be sampled randomly
                within the box instead of an offeset.
            selection_mode: Define how classes should be sampled. 'uniform'
                sampled each object class with the sample probability.
                'sqrt' applies sqrt to the number of objects per class and
                uses those values for weighted sampling.

        Raises:
            ValueError: patch size of dataloder and final patch size need to
                have the same length

        Notes:
            Please refer to the Mixin-Classes for moe details about the
            patch extraction procedure.
        """
        super().__init__(
            data=data,
            batch_size=batch_size,
            patch_size_generator=patch_size_generator,
            patch_size_final=patch_size_final,
            oversample_foreground_percent=oversample_foreground_percent,
            memmap_mode=memmap_mode,
            num_batches_per_epoch=num_batches_per_epoch,
        )
        self.force_bg_case = force_bg_case
        self.offset_prob = offset_prob
        self.offset_magn = offset_magn
        self.max_size_pct = max_size_pct
        self.selection_mode = selection_mode


@DATALOADER_REGISTRY.register
class DataLoader2DOffset(
    RandomBGCrop2D,
    OffsetFGCrop2D,
    RandomSelectionMixin,
    BaseDataLoader2D,
):
    def __init__(
        self,
        data: Dict,
        batch_size: int,
        patch_size_generator: Sequence[int],
        patch_size_final: Sequence[int],
        oversample_foreground_percent: float = 0.5,
        memmap_mode: str = "r+",
        num_batches_per_epoch: int = 2500,
        offset_prob: float = 1.0,
    ):
        """
        Dataloder for 2D Data.
        Center of foreground patches is sampled with an offset while objects
        reamin inside the patch.
        Background patches are sampled randomly.
        Objects are selected randomly.

        Args:
            data: dict with cases and data paths
            batch_size: size of batches to generate
            patch_size_generator: patch size prduced by the dataloader
            patch_size_final: final patch size after spatial transform
            oversample_foreground_percent: Oversample foreground patches.
                Each batch will be balanced to fullfill this criterion.
            memmap_mode: Do not change this. Defaults to "r".
            offset_prob: probability to apply additional offsets of objects.

        Raises:
            ValueError: patch size of dataloder and final patch size need to
                have the same length
        """
        super().__init__(
            data=data,
            batch_size=batch_size,
            patch_size_generator=patch_size_generator,
            patch_size_final=patch_size_final,
            oversample_foreground_percent=oversample_foreground_percent,
            memmap_mode=memmap_mode,
            num_batches_per_epoch=num_batches_per_epoch,
        )
        self.offset_prob = offset_prob


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
class DataLoader3DPOB(DataLoader3DOffsetObjectBalanced):
    @deprecate(
        replacement="`DataLoader3DOffset` with probability parameter",
        deprecate="v0.1.3",
        remove="v0.2",
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@DATALOADER_REGISTRY.register
class DataLoader3DBalanced(DataLoader3DOffsetObjectBalanced):
    @deprecate(
        replacement="`DataLoader3DOffsetBalanced`",
        deprecate="v0.1.3",
        remove="v0.2",
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
