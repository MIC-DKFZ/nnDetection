# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase

import nndet.core.ops_np as ops_np
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
        memmap_mode: str = "r",
        num_batches_per_epoch: int = 2500,
        load_seg: bool = True,
        load_box: bool = False,
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
            load_seg: load segmentation map into `seg` key.
            load_box: load bounding boxes into `target_boxes` and
                `target_classes` key. Working with boxes directly is
                significnatly more efficient in terms of IO and augmentation
                but less accurate during augmentation.

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
        self.load_seg = load_seg
        self.load_box = load_box
        self.save_get_mode = "constant"

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
        instances_batch, properties_batch, case_ids_batch = [], [], []

        if self.load_seg:
            seg_batch = np.zeros(self.seg_shape_batch, dtype=float)
        if self.load_box:
            box_coord_batch = []
            box_label_batch = []

        selected_cases, selected_instances = self.select()
        for batch_idx, (case_id, instance_id) in enumerate(zip(selected_cases, selected_instances)):
            # print(case_id, instance_id)
            case_data = np.load(self._data[case_id]["data_file"], self.memmap_mode, allow_pickle=True)
            properties = load_pickle(self._data[case_id]["properties_file"])

            # determine positions and patches
            if instance_id < 0:
                candidates = self.load_candidates(case_id=case_id, fg_crop=False)
                crop = self.get_bg_crop(
                    case_data=case_data,
                    properties=properties,
                    case_id=case_id,
                    candidates=candidates,
                )
            else:
                candidates = self.load_candidates(case_id=case_id, fg_crop=True)
                crop = self.get_fg_crop(
                    case_data=case_data,
                    properties=properties,
                    case_id=case_id,
                    instance_id=instance_id,
                    candidates=candidates,
                )

            # loading
            data_batch[batch_idx] = save_get_crop(
                case_data,
                crop=crop,
                mode=self.save_get_mode,
                constant_values=0,
            )[0]
            if self.load_seg:
                case_seg = np.load(
                    self._data[case_id]["seg_file"],
                    self.memmap_mode,
                    allow_pickle=True,
                )
                seg_batch[batch_idx] = save_get_crop(
                    case_seg,
                    crop=crop,
                    mode=self.save_get_mode,
                    constant_values=-1,
                )[0]
            if self.load_box:
                res = self.load_box_from_crop(
                    case_id=case_id,
                    case_data=case_data,
                    crop=crop,
                    mode=self.save_get_mode,
                )
                box_coord_batch.append(res[0])
                box_label_batch.append(res[1])
            case_ids_batch.append(case_id)
            instances_batch.append(properties.pop("instances"))
            properties_batch.append(properties)

        out = {
            "data": data_batch,
            "properties": properties_batch,
            "instance_mapping": instances_batch,
            "keys": case_ids_batch,
        }
        if self.load_seg:
            out["seg"] = seg_batch
        if self.load_box:
            out["target_boxes"] = box_coord_batch
            out["target_classes"] = box_label_batch
        return out

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

    def load_box_from_crop(
        self,
        case_id: str,
        case_data: np.ndarray,
        crop: Sequence[slice],
        mode: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load boxes from defined case and crop

        Args:
            case_id: case to load boxes from
            case_data: not used, kept for consistency
            crop: crop of patch
            mode: mode used to load the patch. Refer to `save_get_crop` for
                more information. Only 'constant' mode is supported right now.

        Raises:
            ValueError: Raise if mode is not 'constant'

        Returns:
            Tuple[np.ndarray, np.ndarray]: loaded boxes and labels
                np.ndarray: loaded boxes [N, dim*2]
                np.ndarray: loaded labels [N]
        """
        if mode not in ("constant",):
            raise ValueError(f"Mode {mode} for IO is not supported.")

        gt = np.load(
            self._data[case_id]["label_boxes_file"],
            mmap_mode="r",
            allow_pickle=True,
        )
        gt_boxes = gt["boxes"]
        gt_labels = gt["classes"]

        if gt_boxes.size > 0:
            lower_bound = np.array([s.start for s in crop])
            # offset coordinates to crop
            crop_boxes = gt_boxes - ops_np.expand_to_boxes(lower_bound[None])
            crop_boxes = ops_np.clip_boxes_to_image(crop_boxes, img_shape=self.patch_size_generator)
            # remove small boxes (everything outside of the crop has size 0 or 1)
            keep = ops_np.remove_small_boxes(crop_boxes, min_size=2)

            box_coord = crop_boxes[keep]
            box_label = gt_labels[keep]
        else:
            ndim = case_data.ndim - 1
            box_coord = np.array([[]], dtype=float).reshape(-1, ndim * 2)
            box_label = np.array([], dtype=int)
        return box_coord, box_label


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
        memmap_mode: str = "r",
        num_batches_per_epoch: int = 2500,
        force_bg_case: bool = False,
        offset_prob: float = 1.0,
        offset_magn: float = 1.0,
        load_seg: bool = True,
        load_box: bool = False,
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
            load_seg: load segmentation map into `seg` key.
            load_box: load bounding boxes into `target_boxes` and
                `target_classes` key. Working with boxes directly is
                significnatly more efficient in terms of IO and augmentation
                but less accurate during augmentation.

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
            load_seg=load_seg,
            load_box=load_box,
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
        memmap_mode: str = "r",
        num_batches_per_epoch: int = 2500,
        force_bg_case: bool = False,
        offset_prob: float = 1.0,
        offset_magn: float = 1.0,
        load_seg: bool = True,
        load_box: bool = False,
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
            load_seg: load segmentation map into `seg` key.
            load_box: load bounding boxes into `target_boxes` and
                `target_classes` key. Working with boxes directly is
                significnatly more efficient in terms of IO and augmentation
                but less accurate during augmentation.

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
            load_seg=load_seg,
            load_box=load_box,
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
        memmap_mode: str = "r",
        num_batches_per_epoch: int = 2500,
        force_bg_case: bool = False,
        offset_prob: float = 1.0,
        offset_magn: float = 1.0,
        max_size_pct: float = 1.0,
        load_seg: bool = True,
        load_box: bool = False,
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
            load_seg: load segmentation map into `seg` key.
            load_box: load bounding boxes into `target_boxes` and
                `target_classes` key. Working with boxes directly is
                significnatly more efficient in terms of IO and augmentation
                but less accurate during augmentation.

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
            load_seg=load_seg,
            load_box=load_box,
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
        memmap_mode: str = "r",
        num_batches_per_epoch: int = 2500,
        force_bg_case: bool = False,
        offset_prob: float = 1.0,
        offset_magn: float = 1.0,
        max_size_pct: float = 1.0,
        selection_mode: Union[str, SelectionMode] = "uniform",
        load_seg: bool = True,
        load_box: bool = False,
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
            load_seg: load segmentation map into `seg` key.
            load_box: load bounding boxes into `target_boxes` and
                `target_classes` key. Working with boxes directly is
                significnatly more efficient in terms of IO and augmentation
                but less accurate during augmentation.

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
            load_seg=load_seg,
            load_box=load_box,
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
        memmap_mode: str = "r",
        num_batches_per_epoch: int = 2500,
        force_bg_case: bool = False,
        offset_prob: float = 1.0,
        offset_magn: float = 1.0,
        max_size_pct: float = 1.0,
        selection_mode: Union[str, SelectionMode] = "uniform",
        load_seg: bool = True,
        load_box: bool = False,
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
            load_seg: load segmentation map into `seg` key.
            load_box: load bounding boxes into `target_boxes` and
                `target_classes` key. Working with boxes directly is
                significnatly more efficient in terms of IO and augmentation
                but less accurate during augmentation.

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
            load_seg=load_seg,
            load_box=load_box,
        )
        self.force_bg_case = force_bg_case
        self.offset_prob = offset_prob
        self.offset_magn = offset_magn
        self.max_size_pct = max_size_pct
        self.selection_mode = selection_mode


@DATALOADER_REGISTRY.register
class NoiseLoader(BaseDataLoader3D):
    def __init__(
        self,
        data: Dict,
        batch_size: int,
        patch_size_generator: Sequence[int],
        patch_size_final: Sequence[int],
        oversample_foreground_percent: float = 0.5,
        memmap_mode: str = "r",
        num_batches_per_epoch: int = 2500,
        load_seg: bool = True,
        load_box: bool = False,
        **kwargs,
    ):
        """
        A special dataloader only loading a single (artifically generated)
        cached case.

        Args:
            data: Ignored.
            batch_size: size of batches to generate
            patch_size_generator: Ignored.
            patch_size_final: final patch size after spatial transform
            oversample_foreground_percent: Ignored.
            memmap_mode: Ignored.
            num_batches_per_epoch: number of batcher per epoch
            load_seg: load segmentation map into `seg` key.
            load_box: load bounding boxes into `target_boxes` and
                `target_classes` key. Working with boxes directly is
                significnatly more efficient in terms of IO and augmentation
                but less accurate during augmentation.

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
            load_seg=load_seg,
            load_box=load_box,
        )
        self.data_batch = None
        self.seg_batch = None
        self.box_coords_batch = None
        self.box_labels_batch = None

    def build_cache(self):
        return []

    def __len__(self):
        return self.num_batches_per_epoch

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
        if self.data_batch is None:
            self.data_batch = np.zeros(self.data_shape_batch, dtype=float)
        if self.seg_batch is None:
            self.seg_batch = np.zeros(self.seg_shape_batch, dtype=float)
            self.seg_batch[0, 0, 16:32, 16:32, 16:32] = 1
            self.seg_batch[1, 0, 4:28, 4:28, 4:28] = 1

        batch_size = self.data_shape_batch[0]

        if self.box_coords_batch is None:
            self.box_coords_batch = [
                np.array([[15, 15, 32, 32, 15, 32]]),
                np.array([[3, 3, 28, 28, 3, 28]]),
                *[np.array([[]]).reshape(0, 6) for _ in range(batch_size - 2)],
            ]
            self.box_labels_batch = [
                np.array([0]),
                np.array([0]),
                *[np.array([]) for _ in range(batch_size - 2)],
            ]

        # instances_batch = [{"1": 0} for _ in batch_size]
        instances_batch = [{"1": 0} if idx in [0, 1] else {} for idx in range(batch_size)]
        properties_batch = [{} for _ in range(batch_size)]
        case_ids_batch = ["case_noise" for _ in range(batch_size)]

        batch = {
            "data": self.data_batch,
            "properties": properties_batch,
            "instance_mapping": instances_batch,
            "keys": case_ids_batch,
        }
        if self.load_seg:
            batch["seg"] = self.seg_batch
        if self.load_box:
            batch["target_boxes"] = self.box_coords_batch
            batch["target_classes"] = self.box_labels_batch

        return batch


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


####
# Deprecated -- Do not use this
####
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

            data_batch[batch_idx] = save_get_crop(case_data, crop=crop, mode=self.pad_mode, **self.pad_kwargs_data,)[
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
        memmap_mode: str = "r",
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
