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

from typing import List

from loguru import logger
from collections import OrderedDict
from pathlib import Path

from nndet.io.load import load_pickle
from nndet.io.paths import get_case_ids_from_dir, get_case_id_from_path, Pathlike


def get_np_paths_from_dir(directory: Pathlike) -> List[str]:
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
        Path(directory), remove_modality=False, join=True, pattern="*.npy")
    if not case_paths:
        logger.info(f"Did not find any npy files, looking for npz files. Folder: {directory}")
        case_paths = get_case_ids_from_dir(
            Path(directory), remove_modality=False, join=True, pattern="*.npz")
        if not case_paths:
            logger.error(f"Did not find any npz files.")
            raise RuntimeError(f"Did not find any npz files. Folder: {directory}")
    case_paths = [f for f in case_paths if not f.endswith("_seg")]
    case_paths.sort()
    return case_paths


def load_dataset(folder: Pathlike) -> dict:
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
        dataset[c]['data_file'] = str(folder / f"{c}.npy")
        dataset[c]['seg_file'] = str(folder / f"{c}_seg.npy")
        dataset[c]['properties_file'] = str(folder / f"{c}.pkl")
        dataset[c]['boxes_file'] = str(folder / f"{c}_boxes.pkl")
    return dataset


def load_dataset_id(folder: Pathlike) -> dict:
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
        dataset[c]['data_file'] = str(folder / f"{c}.npy")
        dataset[c]['data_file'] = str(folder / f"{c}.npy")
        dataset[c]['seg_file'] = str(folder / f"{c}_seg.npy")
        dataset[c]['properties_file'] = str(folder / f"{c}.pkl")
        dataset[c]['boxes_file'] = str(folder / f"{c}_boxes.pkl")
    return dataset
