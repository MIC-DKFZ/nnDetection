# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
import os
from collections import OrderedDict
from pathlib import Path
from typing import List

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
    case_paths = get_case_ids_from_dir(
        Path(directory), remove_modality=False, join=True, pattern="*.npy"
    )
    if not case_paths:
        logger.info(
            f"Did not find any npy files, looking for npz files. Folder: {directory}"
        )
        case_paths = get_case_ids_from_dir(
            Path(directory), remove_modality=False, join=True, pattern="*.npz"
        )
        if not case_paths:
            logger.error("Did not find any npz files.")
            raise RuntimeError(f"Did not find any npz files. Folder: {directory}")
    case_paths = [f for f in case_paths if "_seg" not in f]
    case_paths.sort()
    return case_paths


def load_dataset(folder: os.PathLike) -> dict:
    """
    Load dataset (path and properties, NOT the actual data) and
    save them into dict by their path

    Args:
        folder: folder to look for data

    Raises:
        RuntimeError: data needs to be provided in npy or npz format

    Returns:
        dict: loaded data
    """
    folder = Path(folder)
    case_identifiers = get_np_paths_from_dir(folder)

    dataset = OrderedDict()
    for c in case_identifiers:
        dataset[c] = OrderedDict()
        dataset[c]["data_file"] = str(folder / f"{c}.npy")
        dataset[c]["seg_file"] = str(folder / f"{c}_seg.npy")
        dataset[c]["properties_file"] = str(folder / f"{c}.pkl")
        dataset[c]["boxes_file"] = str(folder / f"{c}_boxes.pkl")
    return dataset


def load_dataset_id(folder: os.PathLike) -> dict:
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
    folder = Path(folder)
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
    return dataset
