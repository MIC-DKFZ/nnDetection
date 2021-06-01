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
import pickle
import numpy as np

from loguru import logger
from multiprocessing.pool import Pool
from pathlib import Path
from typing import List, Tuple, Sequence
from scipy.ndimage import binary_fill_holes

from nndet.io.paths import get_case_id_from_path
from nndet.io.load import load_case_from_list


def create_nonzero_mask(data: np.ndarray) -> np.ndarray:
    """
    Create a nonzero mask from data

    Args:
        data (np.ndarray): input data [C, X, Y, Z]

    Returns:
        np.ndarray: binary mask on nonzero regions [X, Y, Z]
    """
    assert len(data.shape) == 4 or len(data.shape) == 3, \
        "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.max(data != 0, axis=0)
    nonzero_mask = binary_fill_holes(nonzero_mask.astype(bool))
    return nonzero_mask


def get_bbox_from_mask(mask: np.ndarray, outside_value: int = 0) -> List[Tuple]:
    """
    Create a bounding box from a mask

    Args:
        mask (np.ndarray): mask [X, Y, Z]
        outside_value (int): background value

    Returns:
        np.ndarray: [(dim0_min, dim0_max), (dim1_min, dim1_max), (dim2_min, dim2_max))
    """
    mask_voxel_coords = (mask != outside_value).nonzero()

    min0idx = int(np.min(mask_voxel_coords[0]))
    max0idx = int(np.max(mask_voxel_coords[0])) + 1

    min1idx = int(np.min(mask_voxel_coords[1]))
    max1idx = int(np.max(mask_voxel_coords[1])) + 1
    idx = [(min0idx, max0idx), (min1idx, max1idx)]

    if len(mask_voxel_coords) == 3:
        min2idx = int(np.min(mask_voxel_coords[2]))
        max2idx = int(np.max(mask_voxel_coords[2])) + 1
        idx.append((min2idx, max2idx))
    return idx


def crop_to_bbox_no_channels(image, bbox: Sequence[Sequence[int]]):
    """
    Crops image to bounding box (in spatial dimensions)

    Args:
        image (arraylike): 2d or 3d array
        bbox (Sequence[Sequence[int]]): bounding box coordinated in an interleaved fashion
            (e.g. (x1, x2), (y1, y2), (z1, z2))

    Returns:
        arraylike: cropped array
    """
    resizer = tuple([slice(_dim[0], _dim[1]) for _dim in bbox])
    return image[resizer]


def crop_to_bbox(data: np.ndarray, bbox: Sequence[Sequence[int]]):
    """
    Crops image to bounding box (performed per channel)

    Args:
        data (np.ndarray): 3d or 4d array [C, X, Y, (Z)]
        bbox (Sequence[Sequence[int]]): bounding box coordinated in an interleaved fashion
            (e.g. (x1, x2), (y1, y2), (z1, z2))

    Returns:
        np.ndarray: cropped array
    """
    cropped_data = []
    for c in range(data.shape[0]):
        cropped = crop_to_bbox_no_channels(data[c], bbox)
        cropped_data.append(cropped)
    data = np.stack(cropped_data)
    return data


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """
    Crop data to nonzero region of data

    Args:
        data (np.ndarray): data to crop
        seg (np.ndarray): segmenation
        nonzero_label (int): nonzero label is written into segmentation map
            where only background was found

    Returns:
        np.ndarray: cropped data
        np.ndarray: cropped and filled (with nonzero_label) segmentation
        List[Tuple[int]]: bounding box of nonzero region
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask, 0)

    data = crop_to_bbox(data, bbox)
    if seg is not None:
        seg = crop_to_bbox(seg, bbox)
    nonzero_mask = crop_to_bbox_no_channels(nonzero_mask, bbox)[None]

    if seg is not None:
        seg[(seg == 0) & (nonzero_mask == 0)] = nonzero_label
    else:
        nonzero_mask = nonzero_mask.astype(np.int32)
        nonzero_mask[nonzero_mask == 0] = nonzero_label
        nonzero_mask[nonzero_mask > 0] = 0
        seg = nonzero_mask
    return data, seg, bbox


class ImageCropper(object):
    def __init__(self, num_processes: int, output_dir: Path = None):
        """
        Helper class to crop images to non zero region (must hold for all modalities)
        In the case of BRaTS and ISLES data this results in a significant reduction in image size

        Args:
            num_processes (int): number of processes to use for cropping
            output_dir (Path): path to output directory
        """
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.num_processes = num_processes
        self.maybe_init_output_dir()

    def maybe_init_output_dir(self):
        """
        Create output directory if  it does not already exists
        """
        if self.output_dir is not None and not self.output_dir.is_dir():
            self.output_dir.mkdir()

    def run_cropping(self, case_files: List[List[Path]], overwrite_existing: bool = False,
                     output_dir: Path = None, copy_gt_data: bool = True):
        """
        Crops data to non zero region and saves them into output_dir
        Optional: also copies ground truth data

        Args:
            case_files (List[List[Path]]): list with all cases in the structure [Case[Case Files]];
                where case files are sorted to corresponding modalities (last file is the label file)
            overwrite_existing (bool): overwrite existing crops
            output_dir (Path): path to output directory
            copy_gt_data (bool): copies ground truth data to output directory
        """
        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.maybe_init_output_dir()

        if copy_gt_data:
            self.copy_gt_data(case_files)

        list_of_args = []
        for _i, case in enumerate(case_files):
            case_id = get_case_id_from_path(str(case[0]))
            assert not case_id.endswith(".gz") and not case_id.endswith(".nii")
            list_of_args.append((case, case_id, overwrite_existing))

        with Pool(processes=self.num_processes) as p:
            p.map(self._process_data_star, list_of_args)

    def copy_gt_data(self, case_files: List[List[Path]]):
        """
        Copy ground truth to output directory
        """
        output_dir_gt = self.output_dir / "labelsTr"
        if not output_dir_gt.is_dir():
            output_dir_gt.mkdir()

        for j, case in enumerate(case_files):
            if case[-1] is not None:
                shutil.copy(case[-1], output_dir_gt)

    def _process_data_star(self, args):
        """
        Unpack argument for function
        """
        return self.process_data(*args)

    def process_data(self, case: List[Path], case_id: str, overwrite_existing: bool = False):
        """
        Extract nonzero region from all cases and create a single array where segmentation
        is located in the last channel and save as npz (saved in key `data`)
        Additional properties per case are saved inside a pkl file

        Args:
            case (List[Path]): list of paths to data and label (label is always at the last position
                and data is sorted after modalities)
            case_id (str): case identifier
            overwrite_existing (bool): overwrite existing data
        """
        try:
            logger.info(f"Processing case {case_id}")
            npz_exists = (self.output_dir / f"{case_id}.npz").is_file()
            pkl_exists = (self.output_dir / f"{case_id}.pkl").is_file()

            if (not npz_exists and not pkl_exists) or overwrite_existing:
                data, seg, properties = self.load_crop_from_list_of_files(case[:-1], case[-1])

                all_data = np.vstack((data, seg))
                np.savez_compressed(self.output_dir / f"{case_id}.npz", data=all_data)
                with open(self.output_dir / f"{case_id}.pkl", 'wb') as f:
                    pickle.dump(properties, f)
            else:
                logger.warning(f"Case {case_id} already exists and overwrite is deactivated")
        except Exception as e:
            logger.info(f"exception in: {case_id}: {e}")
            raise e

    @staticmethod
    def load_crop_from_list_of_files(data_files: List[Path], seg_file: Path = None):
        """
        Load and crop form list of files

        Args:
            data_files (List[Path]): paths to data files
            seg_file (Path): pth to segmentation

        Returns:
            np.ndarray: cropped data
            np.ndarray: cropped (and filled segmentation: -1 where no forground exists) label
            dict: additional properties
                `original_size_of_raw_data`: original shape of data (correctly reordered)
                `original_spacing`: original spacing (correctly reordered)
                `list_of_data_files`: paths of data files
                `seg_file`: path to label file
                `itk_origin`: origin in world coordinates
                `itk_spacing`: spacing in world coordinates
                `itk_direction`: direction in world coordinates
                `crop_bbox`: List[Tuple[int]] cropped bounding box
                `classes`: present classes in segmentation
                `size_after_cropping`: size after cropping
        """
        data, seg, properties = load_case_from_list(data_files, seg_file)
        return ImageCropper.crop(data, properties, seg)

    @staticmethod
    def crop(data: np.ndarray, properties: dict, seg: np.ndarray = None):
        """
        Crop data and segmentation to non zero region

        Args:
            data (np.ndarray): data to crop [C, X, Y, Z]
            properties (dict): additional properties
            seg (np.ndarray): segmentation [1, X, Y, Z]

        Returns:
            data (np.ndarray): data to crop [C, X, Y, Z]
            seg (np.ndarray): segmentation [1, X, Y, Z]
            properties (dict): newly added properties
                `crop_bbox`: List[Tuple[int]] cropped bounding box
                `classes`: present classes in segmentation
                `size_after_cropping`: size after cropping
        """
        shape_before = data.shape
        data, seg, bbox = crop_to_nonzero(data, seg, nonzero_label=-1)
        shape_after = data.shape
        # logger.info(f"Shape before crop {shape_before}; after crop {shape_after}; "
        #             f"spacing {np.array(properties['original_spacing'])}")

        properties["crop_bbox"] = bbox
        properties['classes'] = np.unique(seg)
        seg[seg < -1] = 0
        properties["size_after_cropping"] = data[0].shape
        return data, seg, properties
