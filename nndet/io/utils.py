# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
import os
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

from loguru import logger

from nndet.io.paths import get_case_id_from_path, get_case_ids_from_dir


def get_np_paths_from_dir(directory: os.PathLike) -> List[str]:
    """
    First looks for npz files inside dir. If no files are found, it looks
    for npy files.

    Args:
        directory: path to folder

    Raises:
        RuntimeError: raised if no npy and no npz files are found

    Returns:
        List[str]: paths to files
    """
    case_paths = get_case_ids_from_dir(Path(directory), remove_modality=False, join=True, pattern="*.b2nd")
    if not case_paths:
        logger.info(f"Did not find any b2nd files, looking for npy files. Folder: {directory}")
        case_paths = get_case_ids_from_dir(Path(directory), remove_modality=False, join=True, pattern="*.npy")
        if not case_paths:
            logger.info(f"Did not find any npy files, looking for npz files. Folder: {directory}")
            case_paths = get_case_ids_from_dir(Path(directory), remove_modality=False, join=True, pattern="*.npz")
            if not case_paths:
                logger.error("Did not find any npz files.")
                raise RuntimeError(f"Did not find any npz files. Folder: {directory}")
    case_paths = [f for f in case_paths if not f.endswith("_seg")]
    case_paths.sort()
    return case_paths


def load_dataset_id(data_dir: os.PathLike, label_dir: Optional[os.PathLike] = None) -> dict:
    """
    Load dataset (path and properties, NOT the actual data) and
    save them into dict by their identifier

    Args:
        folder: folder to look for data

    Raises:
        RuntimeError: data needs to be provided in npy or npz format

    Returns:
        dict: loaded data
    """
    folder = Path(data_dir)
    if label_dir is not None:
        label_dir = Path(label_dir)
        assert label_dir.is_dir()

    case_paths = get_np_paths_from_dir(folder)
    case_ids = [get_case_id_from_path(c, remove_modality=False) for c in case_paths]

    dataset = OrderedDict()
    for c in case_ids:
        dataset[c] = OrderedDict()
        dataset[c]["data_file"] = str(folder / f"{c}.npy")
        dataset[c]["data_file"] = str(folder / f"{c}.npy")
        dataset[c]["seg_file"] = str(folder / f"{c}_seg.npy")
        dataset[c]["properties_file"] = str(folder / f"{c}.pkl")
        dataset[c]["boxes_file"] = str(folder / f"{c}_boxes.pkl")

        if label_dir is not None:
            dataset[c]["label_boxes_file"] = str(label_dir / f"{c}_boxes_gt.npz")
            assert Path(dataset[c]["label_boxes_file"]).is_file()
            dataset[c]["label_instances_file"] = str(label_dir / f"{c}_instances_gt.npz")
            assert Path(dataset[c]["label_instances_file"]).is_file()
            dataset[c]["label_seg_file"] = str(label_dir / f"{c}_seg_gt.npz")
            assert Path(dataset[c]["label_seg_file"]).is_file()
    return dataset
