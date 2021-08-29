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

import shutil
import numpy as np
import SimpleITK as sitk

from pathlib import Path
from typing import List, Sequence

from loguru import logger
from sklearn.model_selection import train_test_split

from nndet.io.paths import Pathlike
from nndet.io.paths import get_case_ids_from_dir


__all__ = ["maybe_split_4d_nifti"]


def maybe_split_4d_nifti(source_file: Path, output_folder: Path):
    """
    Process a single nifti file
    if 3D File: copies file to target location
    if 4D File: splits into multiple 3D files and append _0000 ending to indicate channels

    Args:
        source_file (Path): path to source file
        output_folder (Path): path to target directory

    Raises
        TypeError: Data must be 3D or 4D
    """
    img_itk = sitk.ReadImage(str(source_file))
    dim = img_itk.GetDimension()
    filename = source_file.name
    if dim == 3:
        # -7 cuts the .nii.gz part
        shutil.copy(str(source_file), str(output_folder / (filename[:-7] + "_0000.nii.gz")))
        return
    elif dim == 4:
        imgs_splitted = split_4d_itk(img_itk)
        
        for idx, img in enumerate(imgs_splitted):
            sitk.WriteImage(img, str(output_folder / (filename[:-7] + "_%04.0d.nii.gz" % idx)))
    else:
        raise TypeError(f"Unexpected dimensionality: {dim} of file {source_file}, cannot split")


def split_4d_itk(img_itk: sitk.Image) -> List[sitk.Image]:
    """
    Helper function to split 4d itk images into multiple 3 images

    Args:
        img_itk: 4D input image

    Returns:
        List[sitk.Image]: 3d output images
    """
    img_npy = sitk.GetArrayFromImage(img_itk)
    spacing = img_itk.GetSpacing()
    origin = img_itk.GetOrigin()
    direction = np.array(img_itk.GetDirection()).reshape(4, 4)

    spacing = tuple(list(spacing[:-1]))
    assert len(spacing) == 3
    origin = tuple(list(origin[:-1]))
    assert len(origin) == 3
    direction = tuple(direction[:-1, :-1].reshape(-1))
    assert len(direction) == 9

    images_new = []
    for i, t in enumerate(range(img_npy.shape[0])):
            img = img_npy[t]
            images_new.append(
                create_itk_image_spatial_props(img, spacing, origin, direction))
    return images_new


def create_itk_image_spatial_props(
        data: np.ndarray, spacing: Sequence[float], origin: Sequence[float],
        direction: Sequence[Sequence[float]]) -> sitk.Image:
    """
    Create new sitk image and set spatial tags

    Args:
        data: data
        spacing: spacing
        origin: origin
        direction: directiont

    Returns:
        sitk.Image: new image
    """
    data_itk = sitk.GetImageFromArray(data)
    data_itk.SetSpacing(spacing)
    data_itk.SetOrigin(origin)
    data_itk.SetDirection(direction)
    return data_itk


def create_test_split(splitted_dir: Pathlike,
                      num_modalities: int,
                      test_size: float = 0.3,
                      random_state: int = 0,
                      shuffle: bool = True,
                      ):
    """
    Helper function to create an artificial test split from the splitted data

    Args:
        splitted_dir: path to directory with splitted data. `imagesTr` and
            `labelsTr` need to exist beforehand. `imagesTs` and `labelsTs`
            will be created automatically.
        num_modalities: number of modalities
        test_size: size of test set, needs to be a value between 0 and 1
        seed: seed for splitting
        shuffle: shuffle data
    """
    images_tr = Path(splitted_dir) / "imagesTr"
    labels_tr = Path(splitted_dir) / "labelsTr"
    images_ts = Path(splitted_dir) / "imagesTs"
    labels_ts = Path(splitted_dir) / "labelsTs"

    if not images_tr.is_dir():
        raise ValueError(f"No dir with training images found {images_tr}")
    if not labels_tr.is_dir():
        raise ValueError(f"No dir with training labels found {labels_tr}")
    images_ts.mkdir(parents=True, exist_ok=True)
    labels_ts.mkdir(parents=True, exist_ok=True)

    case_ids = sorted(get_case_ids_from_dir(images_tr, remove_modality=True))
    logger.info(f"Found {len(case_ids)} to split")

    train_ids, test_ids = train_test_split(
        case_ids, test_size=test_size, random_state=random_state, shuffle=shuffle)
    logger.info(f"Using {train_ids} for training and {test_ids} for testing.")

    for cid in test_ids:
        for modality in range(num_modalities):
            shutil.move(images_tr / f"{cid}_{modality:04d}.nii.gz",
                        images_ts / f"{cid}_{modality:04d}.nii.gz")
        shutil.move(labels_tr / f"{cid}.nii.gz", labels_ts / f"{cid}.nii.gz")
        if (labels_tr / f"{cid}.json").is_file():
            shutil.move(labels_tr / f"{cid}.json", labels_ts / f"{cid}.json")
