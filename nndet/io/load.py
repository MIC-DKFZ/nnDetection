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

import os
import pickle
import json
import yaml
import time
from contextlib import contextmanager
from itertools import repeat
from multiprocessing.pool import Pool
from collections import OrderedDict
from pathlib import Path
from typing import Sequence, Any, Tuple, Union
from zipfile import BadZipfile

import numpy as np
import SimpleITK as sitk
from loguru import logger

from nndet.io.paths import subfiles, Pathlike


__all__ = ["load_case_cropped", "load_case_from_list",
           "load_properties_of_cropped", "npy_dataset",
           "load_pickle", "load_json", "save_json", "save_pickle",
           "save_yaml", "load_npz_looped",
           ]


def load_case_from_list(data_files, seg_file=None) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load data and label of one case from list of paths

    Args:
        data_files (Sequence[Path]): paths to data files
        seg_file (Path): path to segmentation file (if a second file
            with a json ending is found, it is treated as an additional
            property file and will be loaded automatically)

    Returns:
        np.ndarary: loaded data (as float32) [C, X, Y, Z]
        np.ndarray: loaded segmentation (if no segmentation was provided, None)
            (as float32) [1, X, Y, Z]
        dict: additional properties of files
            `original_size_of_raw_data`: original shape of data (correctly reordered)
            `original_spacing`: original spacing (correctly reordered)
            `list_of_data_files`: paths of data files
            `seg_file`: path to label file
            `itk_origin`: origin in world coordinates
            `itk_spacing`: spacing in world coordinates
            `itk_direction`: direction in world coordinates
    """
    assert isinstance(data_files, Sequence), "case must be sequence"
    properties = OrderedDict()
    data_itk = [sitk.ReadImage(str(f)) for f in data_files]

    properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[[2, 1, 0]]
    properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
    properties["list_of_data_files"] = data_files
    properties["seg_file"] = seg_file

    properties["itk_origin"] = data_itk[0].GetOrigin()
    properties["itk_spacing"] = data_itk[0].GetSpacing()
    properties["itk_direction"] = data_itk[0].GetDirection()

    data_npy = np.stack([sitk.GetArrayFromImage(d) for d in data_itk])
    if seg_file is not None:
        seg_itk = sitk.ReadImage(str(seg_file))
        seg_npy = sitk.GetArrayFromImage(seg_itk)[None].astype(np.float32)

        seg_props_file = f"{str(seg_file).split('.')[0]}.json"
        if os.path.isfile(seg_props_file):
            properties_json = load_json(seg_props_file)

            # cast instances to correct type
            properties_json["instances"] = {
                str(key): int(item) for key, item in properties_json["instances"].items()}

            properties.update(properties_json)
    else:
        seg_npy = None
    return data_npy.astype(np.float32), seg_npy, properties


def load_properties_of_cropped(path: Path):
    """
    Load property file of after cropping was performed
    (files are name after case id and .pkl ending)
    
    Args:
        path (Path): path to file (if .pkl is missing, it will be added automatically)
    
    Returns:
        Dict: loaded properties
    """
    if not path.suffix == '.pkl':
        path = Path(str(path) + '.pkl')
    
    with open(path, 'rb') as f:
        properties = pickle.load(f)
    return properties


def load_case_cropped(folder: Path, case_id: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load single case after cropping
    
    Args:
        folder (Path): path to folder where cases are located
        case_id (str): case identifier
    
    Returns:
        np.ndarray: data
        np.ndarray: segmentation
        dict: additional properties
    """
    stack = load_npz_looped(os.path.join(folder, case_id) + ".npz",
                            keys=["data"], num_tries=3,
                            )["data"]
    data = stack[:-1]
    seg = stack[-1]

    with open(os.path.join(folder, case_id) + ".pkl", "rb") as f:
        props = pickle.load(f)
    assert data.shape[1:] == seg.shape, (f"Data and segmentation need to have same dim (except first). "
                                         f"Found data {data.shape} and "
                                         f"mask {seg.shape} for case {case_id}")
    return data.astype(np.float32), seg.astype(np.int32), props


@contextmanager
def npy_dataset(folder: str, processes: int,
                unpack: bool = True, delete_npy: bool = True,
                delete_npz: bool = False):
    """
    Automatically unpacks the npz dataset and deletes npy data after completion

    Args:
        folder: path to folder
        processes: number of processes to use
        unpack: unpack data
        delete_npy: delete npy files at the end
        delete_npz: delete the npz file after conversion
    """
    if unpack:
        unpack_dataset(Path(folder), processes, delete_npz=delete_npz)
    try:
        yield True
    finally:
        if delete_npy:
            del_npy(Path(folder))


def unpack_dataset(folder: Pathlike,
                   processes: int,
                   delete_npz: bool = False):
    """
    unpacks all npz files in a folder to npy
    (whatever you want to have unpacked must be saved under key)

    Args
        folder: path to folder where data is located
        processes: number of processes to use
        key: key which should be extracted
        delete_npz: delete the npz file after conversion
    """
    logger.info("Unpacking dataset")
    npz_files = subfiles(Path(folder), identifier="*.npz", join=True)
    if not npz_files:
        logger.warning(f'No paths found in {Path(folder)} matching *.npz')
        return
    with Pool(processes) as p:
        p.starmap(npz2npy, zip(npz_files, repeat(delete_npz)))


def pack_dataset(folder, processes: int, key: str):
    """
    Pack dataset (from npy to npz)

    Args
        folder: path to folder where data is located
        processes: number of processes to use
        key: key which should be extracted
    """
    logger.info("Packing dataset")
    npy_files = subfiles(Path(folder), identifier="*.npy", join=True)
    with Pool(processes) as p:
        p.starmap(npy2npz, zip(npy_files, repeat(key)))


def npz2npy(npz_file: str, delete_npz: bool = False):
    """
    convert npz to npy

    Args:
        npz_file: path to npz file
        delete_npz: delete the npz file after conversion
    """
    if not os.path.isfile(npz_file[:-3] + "npy"):
        a = load_npz_looped(npz_file, keys=["data", "seg"], num_tries=3)
        if a is not None:
            np.save(npz_file[:-3] + "npy", a["data"])
            np.save(npz_file[:-4] + "_seg.npy", a["seg"])
    if delete_npz:
        os.remove(npz_file)


def npy2npz(npy_file: str, key: str):
    """
    convert npy to npz

    Args:
        npy_file: path to npy file
        key: key to extract
    """
    d = np.load(npy_file)
    np.savez_compressed(npy_file[:-3] + "npz", **{key: d})


def del_npy(folder: Pathlike):
    """
    Deletes all npy files inside folder
    """
    npy_files = Path(folder).glob("*.npy")
    npy_files = [i for i in npy_files if os.path.isfile(i)]
    logger.info(f"Found {len(npy_files)} for removal")
    for n in npy_files:
        os.remove(n)


def load_json(path: Path, **kwargs) -> Any:
    """
    Load json file

    Args:
        path: path to json file
        **kwargs: keyword arguments passed to :func:`json.load`

    Returns:
        Any: json data
    """
    if isinstance(path, str):
        path = Path(path)
    if not(".json" == path.suffix):
        path = str(path) + ".json"

    with open(path, "r") as f:
        data = json.load(f, **kwargs)
    return data


def save_json(data: Any, path: Pathlike, indent: int = 4, **kwargs):
    """
    Load json file

    Args:
        data: data to save to json
        path: path to json file
        indent: passed to json.dump
        **kwargs: keyword arguments passed to :func:`json.dump`
    """
    if isinstance(path, str):
        path = Path(path)
    if not(".json" == path.suffix):
        path = Path(str(path) + ".json")

    with open(path, "w") as f:
        json.dump(data, f, indent=indent, **kwargs)


def load_pickle(path: Path, **kwargs) -> Any:
    """
    Load pickle file

    Args:
        path: path to pickle file
        **kwargs: keyword arguments passed to :func:`pickle.load`

    Returns:
        Any: json data
    """
    if isinstance(path, str):
        path = Path(path)
    if not any([fix == path.suffix for fix in [".pickle", ".pkl"]]):
        path = Path(str(path) + ".pkl")

    with open(path, "rb") as f:
        data = pickle.load(f, **kwargs)
    return data


def save_pickle(data: Any, path: Pathlike, **kwargs):
    """
    Load pickle file

    Args:
        data: data to save to pickle
        path: path to pickle file
        **kwargs: keyword arguments passed to :func:`pickle.dump`
    """
    if isinstance(path, str):
        path = Path(path)
    if not any([fix == path.suffix for fix in [".pickle", ".pkl"]]):
        path = str(path) + ".pkl"

    with open(str(path), "wb") as f:
        data = pickle.dump(data, f, **kwargs)
    return data


def save_yaml(data: Any, path: Path, **kwargs):
    """
    Load yaml file

    Args:
        data: data to save to yaml
        path: path to yaml file
        **kwargs: keyword arguments passed to :func:`yaml.dump`
    """
    if isinstance(path, str):
        path = Path(path)
    if not(".yaml" == path.suffix):
        path = str(path) + ".yaml"

    with open(path, "w") as f:
        yaml.dump(data, f, **kwargs)


def save_txt(data: str, path: Path, **kwargs):
    """
    Load yaml file

    Args:
        data: data to save to txt
        path: path to txt file
        **kwargs: keyword arguments passed to :func:`json.dump`
    """
    if isinstance(path, str):
        path = Path(path)
    if not(".txt" == path.suffix):
        path = str(path) + ".txt"

    with open(path, "a") as f:
        f.write(str(data))


def load_npz_looped(
        p: Pathlike,
        keys: Sequence[str],
        *args,
        num_tries: int = 3,
        **kwargs,
        ) -> Union[np.ndarray, dict]:
    """
    Try | Except loop to load numpy files
    (especially large numpy files can fail with BadZipFile Errors)

    Args:
        p: path to file to load
        keys: keys to load from npz file
        num_tries: number of tries to load file
        *args: passed to `np.load`
        **kwargs: passed to `np.load`

    Returns:
        dict: loaded data
    """
    if num_tries <= 0:
        raise ValueError(f"Num tires needs to be larger than 0, found {num_tries} tries.")

    for i in range(num_tries):  # try reading the file 3 times
        try:
            _data = np.load(str(p), *args, **kwargs)
            data = {k: _data[k] for k in keys}
            break
        except Exception as e:
            if i == num_tries - 1:
                logger.error(f"Could not unpack {p}")
                return None
            time.sleep(5.)
    return data
