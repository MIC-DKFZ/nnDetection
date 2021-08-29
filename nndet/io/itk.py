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

import numpy as np
import SimpleITK as sitk
from itertools import product


from typing import Sequence, Union, Tuple


def create_circle_mask_itk(image_itk: sitk.Image,
                           world_centers: Sequence[Sequence[float]],
                           world_rads: Sequence[float],
                           ndim: int = 3,
                           ) -> sitk.Image:
    """
    Creates an itk image with circles defined by center points and radii

    Args:
        image_itk: original image (used for the coordinate frame)
        world_centers: Sequence of center points in world coordiantes (x, y, z)
        world_rads: Sequence of radii to use
        ndim: number of spatial dimensions

    Returns:
        sitk.Image: mask with circles
    """
    image_np = sitk.GetArrayFromImage(image_itk)
    min_spacing = min(image_itk.GetSpacing())

    if image_np.ndim > ndim:
        image_np = image_np[0]
    mask_np = np.zeros_like(image_np).astype(np.uint8)

    for _id, (world_center, world_rad) in enumerate(zip(world_centers, world_rads), start=1):
        check_rad = (world_rad / min_spacing) * 1.5  # add some buffer to it
        bounds = []
        center = image_itk.TransformPhysicalPointToContinuousIndex(world_center)[::-1]
        for ax, c in enumerate(center):
            bounds.append((
                max(0, int(c - check_rad)),
                min(mask_np.shape[ax], int(c + check_rad)),
            ))
        coord_box = product(*[list(range(b[0], b[1])) for b in bounds])

        # loop over every pixel position
        for coord in coord_box:
            world_coord = image_itk.TransformIndexToPhysicalPoint(tuple(reversed(coord)))  # reverse order to x, y, z for sitk
            dist = np.linalg.norm(np.array(world_coord) - np.array(world_center))
            if dist <= world_rad:
                mask_np[tuple(coord)] = _id
        assert mask_np.max() == _id

    mask_itk = sitk.GetImageFromArray(mask_np)
    mask_itk.SetOrigin(image_itk.GetOrigin())
    mask_itk.SetDirection(image_itk.GetDirection())
    mask_itk.SetSpacing(image_itk.GetSpacing())
    return mask_itk


def load_sitk(path: Union[Path, str], **kwargs) -> sitk.Image:
    """
    Functional interface to load image with sitk

    Args:
        path: path to file to load

    Returns:
        sitk.Image: loaded sitk image
    """
    return sitk.ReadImage(str(path), **kwargs)


def load_sitk_as_array(path: Union[Path, str], **kwargs) -> Tuple[np.ndarray, dict]:
    """
    Functional interface to load sitk image and convert it to an array

    Args:
        path: path to file to load

    Returns:
        np.ndarray: loaded image data
        dict: loaded meta data
    """
    img_itk = load_sitk(path, **kwargs)
    meta = {key: img_itk.GetMetaData(key) for key in img_itk.GetMetaDataKeys()}
    return sitk.GetArrayFromImage(img_itk), meta
