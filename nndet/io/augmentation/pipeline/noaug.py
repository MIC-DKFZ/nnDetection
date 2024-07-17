# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0


from typing import List, Sequence

import numpy as np
from batchgenerators.transforms.channel_selection_transforms import (
    DataChannelSelectionTransform,
    SegChannelSelectionTransform,
)
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform
from batchgenerators.transforms.utility_transforms import (
    NumpyToTensor,
    RemoveLabelTransform,
    RenameTransform,
)
from loguru import logger

from nndet.io.augmentation import AUGMENTATION_REGISTRY
from nndet.io.augmentation.base import AugmentationSetup, ComposePretty, get_patch_size


@AUGMENTATION_REGISTRY.register
class NoAug(AugmentationSetup):
    def __init__(
        self,
        patch_size: Sequence[int],
        params: dict,
        use_box_io: bool = False,
    ) -> None:
        super().__init__(patch_size, params, use_box_io=use_box_io)
        self.dummy_2d = self.params.get("dummy_2D", False)
        if self.dummy_2d:
            logger.info("Running dummy 2d augmentation transforms!")

        if self.dummy_2d:
            self._spatial_transform_patch_size = self.patch_size[1:]
        else:
            self._spatial_transform_patch_size = self.patch_size

    def any_matching_axes(self) -> bool:
        """
        Check if any axes have the same size

        Returns:
            bool: `True` if at least two axes have the same size.
                `False` otherwise
        """
        num_matching_axes = np.array([sum([i == j for j in self.patch_size]) for i in self.patch_size])
        return np.any(num_matching_axes > 1)

    def same_axes(self) -> List[int]:
        """
        Compute number of matching axes of patch size

        Returns:
            List[int: indices of axes which has the same patch size
        """
        num_matching_axes = np.array([sum([i == j for j in self.patch_size]) for i in self.patch_size])
        same_axes = list(np.where(num_matching_axes == np.max(num_matching_axes))[0])
        return same_axes

    def get_patch_size_generator(self) -> List[int]:
        """
        Compute patch size to extract from volume to avoid augmentation
        artifacts
        """
        _patch_size = list(
            get_patch_size(
                patch_size=self._spatial_transform_patch_size,
                rot_x=self.params["rotation_x"],
                rot_y=self.params["rotation_y"],
                rot_z=self.params["rotation_z"],
                scale_range=self.params["scale_range"],
            )
        )
        if self.dummy_2d:
            _patch_size = [self.patch_size[0]] + _patch_size
        return _patch_size

    def get_training_transforms(self):
        tr_transforms = []
        if self.params.get("selected_data_channels"):
            tr_transforms.append(DataChannelSelectionTransform(self.params.get("selected_data_channels")))
        if self.params.get("selected_seg_channels"):
            tr_transforms.append(SegChannelSelectionTransform(self.params.get("selected_seg_channels")))
        tr_transforms.append(CenterCropTransform(self.patch_size))
        tr_transforms.append(RemoveLabelTransform(-1, 0))
        tr_transforms.append(RenameTransform("seg", "target", True))
        tr_transforms.append(NumpyToTensor(["data", "target"], "float"))
        return ComposePretty(tr_transforms)

    def get_validation_transforms(self):
        val_transforms = []
        if self.params.get("selected_data_channels"):
            val_transforms.append(DataChannelSelectionTransform(self.params.get("selected_data_channels")))
        if self.params.get("selected_seg_channels"):
            val_transforms.append(SegChannelSelectionTransform(self.params.get("selected_seg_channels")))
        val_transforms.append(CenterCropTransform(self.patch_size))
        val_transforms.append(RemoveLabelTransform(-1, 0))
        val_transforms.append(RenameTransform("seg", "target", True))
        val_transforms.append(NumpyToTensor(["data", "target"], "float"))
        return ComposePretty(val_transforms)


@AUGMENTATION_REGISTRY.register
class NoAugV2(AugmentationSetup):
    def __init__(
        self,
        patch_size: Sequence[int],
        params: dict,
        use_box_io: bool = False,
    ) -> None:
        super().__init__(patch_size, params, use_box_io=use_box_io)
        self.dummy_2d = self.params.get("dummy_2D", False)
        if self.dummy_2d:
            logger.info("Running dummy 2d augmentation transforms!")

        if self.dummy_2d:
            self._spatial_transform_patch_size = self.patch_size[1:]
        else:
            self._spatial_transform_patch_size = self.patch_size

        if self.use_box_io:
            raise NotImplementedError

    def any_matching_axes(self) -> bool:
        """
        Check if any axes have the same size

        Returns:
            bool: `True` if at least two axes have the same size.
                `False` otherwise
        """
        num_matching_axes = np.array([sum([i == j for j in self.patch_size]) for i in self.patch_size])
        return np.any(num_matching_axes > 1)

    def same_axes(self) -> List[int]:
        """
        Compute number of matching axes of patch size

        Returns:
            List[int: indices of axes which has the same patch size
        """
        num_matching_axes = np.array([sum([i == j for j in self.patch_size]) for i in self.patch_size])
        same_axes = list(np.where(num_matching_axes == np.max(num_matching_axes))[0])
        return same_axes

    def get_patch_size_generator(self) -> List[int]:
        """
        Compute patch size to extract from volume to avoid augmentation
        artifacts
        """
        # TODO : recheck
        _patch_size = list(
            get_patch_size(
                patch_size=self._spatial_transform_patch_size,
                rot_x=self.params["rotation_x"],
                rot_y=self.params["rotation_y"],
                rot_z=self.params["rotation_z"],
                scale_range=self.params["scale_range"],
            )
        )
        if self.dummy_2d:
            _patch_size = [self.patch_size[0]] + _patch_size
        return _patch_size

    def get_training_transforms(self):
        raise NotImplementedError

    def get_validation_transforms(self):
        val_transforms = []
        if self.params.get("selected_data_channels"):
            val_transforms.append(DataChannelSelectionTransform(self.params.get("selected_data_channels")))
        if self.params.get("selected_seg_channels"):
            val_transforms.append(SegChannelSelectionTransform(self.params.get("selected_seg_channels")))
        val_transforms.append(CenterCropTransform(self.patch_size))
        val_transforms.append(RemoveLabelTransform(-1, 0))
        val_transforms.append(RenameTransform("seg", "target", True))
        val_transforms.append(NumpyToTensor(["data", "target"], "float"))
        return ComposePretty(val_transforms)
