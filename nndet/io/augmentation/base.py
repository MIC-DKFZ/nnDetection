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

from typing import Sequence, List
from abc import ABC, abstractmethod

import numpy as np


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

    rot_x = min(90 / 360 * 2. * np.pi, rot_x)
    rot_y = min(90 / 360 * 2. * np.pi, rot_y)
    rot_z = min(90 / 360 * 2. * np.pi, rot_z)

    from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d
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
    def __init__(self, 
                 patch_size: Sequence[int],
                 params: dict,
                 ) -> None:
        """
        Helper class for augmenation setup

        Args:
            patch_size: output patch size of augmentations
            params: augmentation parameters
        
        Notes:
            The needed keys of :attr:`params` depend on the exact
            transformations which should be used. 
        """
        self.patch_size = patch_size
        self.params = params

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
        return list(get_patch_size(
            patch_size=self.patch_size,
            rot_x=self.params['rotation_x'],
            rot_y=self.params['rotation_y'],
            rot_z=self.params['rotation_z'],
            scale_range=self.params['scale_range'],
        ))
