# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np
import torch
from batchgenerators.transforms.abstract_transforms import Compose
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from threadpoolctl import threadpool_limits


class ComposePretty(Compose):
    def __str__(self) -> str:
        s = "--- Augmentation ---\n"
        for tr in self.transforms:
            s += f"{tr}\n"
        s += "---"
        return s


class ComposeBG2(BasicTransform):
    def __init__(self, transforms: List[BasicTransform]):
        """
        This is a custom compose class for batchgeneratorsv2. It iterates
        through a batched data dictionary and applies the transforms to each
        image in the batch.
        Please note that the input of the segmentation is to be expected
        at the 'seg' key while the output will be saved in 'target'.

        Args:
            transforms: transforms to apply to samples (only transforms from
                BGV2 supported here!)
        """
        super().__init__()
        self.transforms = transforms

    def apply(self, data_dict, **params) -> dict:
        """
        Apply transforms to data dictionary

        Args:
            data_dict: dictionary containing information from a batch of data

        Returns:
            dict: transformed batch
        """
        image = []
        segmentation = []

        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for i in range(len(data_dict["data"])):  # iterate over all images in the batch
                    data_dict["image"] = torch.from_numpy(data_dict["data"][i]).to(dtype=torch.float)
                    data_dict["segmentation"] = torch.from_numpy(data_dict["seg"][i]).to(dtype=torch.int16)

                    # iterate over all transforms
                    for t in self.transforms:
                        data_dict = t(**data_dict)

                    image.append(data_dict["image"])
                    segmentation.append(data_dict["segmentation"])
        data_dict["data"] = torch.stack(image)
        data_dict["target"] = torch.stack(segmentation)
        return data_dict

    def __str__(self) -> str:
        s = "--- Augmentation ---\n"
        for tr in self.transforms:
            s += f"{tr}\n"
        s += "---"
        return s


def get_patch_size(
    patch_size: Sequence[int],
    rot_x: float,
    rot_y: float,
    rot_z: float,
    scale_range: Sequence[float],
) -> np.ndarray:
    """
    Compute enlarged patch size for augmentations to reduce
    artifacts at the borders before final cropping

    Args:
        final_patch_size: target spatial size after final cropping
        rot_x: rotation in x in radian
        rot_y: rotation in y in radian
        rot_z: rotation in z in radian
        scale_range: scaling range

    Returns:
        np.ndarray: enlarged patch size for augmentation
    """
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))

    rot_x = min(90 / 360 * 2.0 * np.pi, rot_x)
    rot_y = min(90 / 360 * 2.0 * np.pi, rot_y)
    rot_z = min(90 / 360 * 2.0 * np.pi, rot_z)

    from batchgenerators.augmentations.utils import rotate_coords_2d, rotate_coords_3d

    coords = np.array(patch_size)
    final_shape = np.copy(coords)
    if len(coords) == 3:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
    elif len(coords) == 2:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)
    final_shape /= min(scale_range)
    return final_shape.astype(np.int32)


class AugmentationSetup(ABC):
    def __init__(
        self,
        patch_size: Sequence[int],
        params: dict,
        use_box_io: bool = False,
    ) -> None:
        """
        Helper class for augmenation setup

        Args:
            patch_size: output patch size of augmentations
            params: augmentation parameters
            use_box_io: if `True` augmentation are performed on point based
                representations

        Notes:
            The needed keys of :attr:`params` depend on the exact
            transformations which should be used.
        """
        self.patch_size = patch_size
        self.params = params
        self.use_box_io = use_box_io

    @abstractmethod
    def get_training_transforms(self):
        """
        Setup training transformations
        Needs to be overwritten in subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def get_validation_transforms(self):
        """
        Setup validation transformations
        Needs to be overwritten in subclasses.
        """
        raise NotImplementedError

    def get_patch_size_generator(self) -> List[int]:
        """
        Compute patch size to extract from volume to avoid augmentation
        artifacts
        """
        return list(
            get_patch_size(
                patch_size=self.patch_size,
                rot_x=self.params["rotation_x"],
                rot_y=self.params["rotation_y"],
                rot_z=self.params["rotation_z"],
                scale_range=self.params["scale_range"],
            )
        )
