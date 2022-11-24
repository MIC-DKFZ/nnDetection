# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
# SPDX-License-Identifier: Apache-2.0


import functools
import os
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Optional, Union

import numpy as np
import SimpleITK as sitk

from nndet.io import load_json, load_sitk
from nndet.io.paths import get_task, get_paths_from_splitted_dir
from nndet.utils.config import load_dataset_info
from nndet.utils.info import maybe_verbose_iterable


def env_guard(func):
    """
    Contextmanager to check nnDetection environment variables
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # we use print here because logging might not be initialized yet and
        # this is intended as a user warning.
        
        # det_data
        if os.environ.get("det_data", None) is None:
            raise RuntimeError(
                "'det_data' environment variable not set. "
                "Please refer to the installation instructions. "
                )

        # det_models
        if os.environ.get("det_models", None) is None:
            raise RuntimeError(
                "'det_models' environment variable not set. "
                "Please refer to the installation instructions. "
                )

        # OMP_NUM_THREADS
        if os.environ.get("OMP_NUM_THREADS", None) is None:
            raise RuntimeError(
                "'OMP_NUM_THREADS' environment variable not set. "
                "Please refer to the installation instructions. "
                )

        # det_num_threads
        if os.environ.get("det_num_threads", None) is None:
            warnings.warn(
                "Warning: 'det_num_threads' environment variable not set. "
                "Please read installation instructions again. "
                "Training will not work properly.")

        # det_verbose
        if os.environ.get("det_verbose", None) is None:
            print("'det_verbose' environment variable not set. "
                  "Continue in verbose mode.")

        return func(*args, **kwargs)
    return wrapper


def _check_key_missing(cfg: dict, key: str, ktype=None):
    if key not in cfg:
        raise ValueError(f"Dataset information did not contain "
                        f"'{key}' key, found {list(cfg.keys())}")
    
    if ktype is not None:
        if not isinstance(cfg[key], ktype):
            raise ValueError(f"Found {key} of type {type(cfg[key])} in "
                             f"dataset information but expected type {ktype}")


def check_dataset_file(task_name: str):
    """
    Run a sequence of checks to confirm correct format of dataset information

    Args:
        task_name: task identifier to check info for
    """
    print("Start dataset info check.")
    cfg = load_dataset_info(get_task(task_name))
    _check_key_missing(cfg, "task", ktype=str)
    _check_key_missing(cfg, "dim", ktype=int)
    _check_key_missing(cfg, "labels", ktype=dict)
    _check_key_missing(cfg, "modalities", ktype=dict)

    # check dim
    if dim := cfg["dim"] not in [2, 3]:
        raise ValueError(f"Found dim {dim} in dataset info but only support dim=2 or dim=3.")

    # check labels
    for key, item in cfg["labels"].items():
        if not isinstance(key, str):
            raise ValueError("Expected key of type string in dataset "
                             f"info labels but found {type(key)} : {key}")
        if not isinstance(item, str):
            raise ValueError("Expected name of type string in dataset "
                             f"info labels but found {type(item)} : {item}")
    found_classes = sorted(list(map(int, cfg["labels"].keys())))
    for ic, idx in enumerate(found_classes):
        if ic != idx:
            raise ValueError("Found wrong order of label classes in dataset info."
                             f"Found {found_classes} but expected {list(range(len(found_classes)))}")

    # check modalities
    for key, item in cfg["modalities"].items():
        if not isinstance(key, str):
            raise ValueError("Expected key of type string in dataset "
                             f"info labels but found {type(key)} : {key}")
        if not isinstance(item, str):
            raise ValueError("Expected name of type string in dataset "
                             f"info labels but found {type(item)} : {item}")
    found_mods = sorted(list(map(int, cfg["modalities"].keys())))
    for ic, idx in enumerate(found_classes):
        if ic != idx:
            raise ValueError("Found wrong order of modalities in dataset info."
                             f"Found {found_mods} but expected {list(range(len(found_mods)))}")

    # check target class
    target_class = cfg.get("target_class", None)
    if target_class is not None and not isinstance(target_class, int):
        raise ValueError("If target class is defined, it needs to be an integer, "
                         f"found {type(target_class)} : {target_class}")

    print("Dataset info check complete.")


def check_data_and_label_splitted(
    task_name: str,
    test: bool = False,
    labels: bool = True,
    full_check: bool = True,
    ):
    """
    Perform checks of data and label in raw splitted format

    Args:
        task_name: name of task to check
        test: check test data
        labels: check labels
        full_check: Per default a full check will be performed which needs to
            load all files. If this is disabled, a computationall light check
            will be performed 

    Raises:
        ValueError: if not all raw splitted files were found
        ValueError: missing label info file
        ValueError: instances in label info file need to start at 1
        ValueError: instances in label info file need to be consecutive
    """
    print(f"Start data and label check: test={test}")
    cfg = load_dataset_info(get_task(task_name))

    splitted_paths = get_paths_from_splitted_dir(
        num_modalities=len(cfg["modalities"]),
        splitted_4d_output_dir=Path(os.getenv('det_data')) / task_name / "raw_splitted",
        labels=labels,
        test=test,
    )

    for case_paths in maybe_verbose_iterable(splitted_paths):
        # check all files exist
        for cp in case_paths:
            if not Path(cp).is_file():
                raise ValueError(f"Expected {cp} to be a raw splitted "
                                 "data path but it does not exist.")
            if "." in str(cp.parent):
                raise ValueError(f"Avoid '.' in paths since this confuses nnDetection in its current version, found {str(cp)}")

        if labels:
            # check label info (json files)
            mask_path = case_paths[-1]
            mask_info_path = mask_path.parent / f"{mask_path.stem.split('.')[0]}.json"
            if not Path(mask_info_path).is_file():
                raise ValueError(f"Expected {mask_info_path} to be a raw splitted "
                                "mask info path but it does not exist.")
            mask_info = load_json(mask_info_path)

            _type_check_instances_json(mask_info, mask_info_path, expected_labels=cfg["labels"])

            # check presence / absence of instances in json and mask
            if mask_info["instances"]:
                mask_info_instances = list(map(int, mask_info["instances"].keys()))

                if j := not min(mask_info_instances) == 1:
                    raise ValueError(f"Instance IDs need to start at 1, found {j} in {mask_info_path}")

                for i in range(1, len(mask_info_instances) + 1):
                    if i not in mask_info_instances:
                        raise ValueError(f"Exptected {i} to be an Instance ID in "
                                        f"{mask_info_path} but only found {mask_info_instances}")
        else:
            mask_info_path = None

        if full_check:
            _full_check(case_paths, mask_info_path)
    print("Data and label check complete.")


def _type_check_instances_json(mask_info: Dict, mask_info_path: Union[str, Path], expected_labels: list):
    """
    Check types of json files

    Args:
        mask_info: contains information loaded from the label json file.
            Specifically the `instances` key is checked for a "str":"int" type
        mask_info_path: path to json file where information was loaded from
        expected_labels: list with the expected labels

    Raises:
        ValueError: raised if instance ids are not typed as str
        ValueError: raised if instance classes are not typed as int
    """
    # transform expected labels to integer
    exp_labs = [int(lab) for lab in expected_labels]
    # type check instances key
    for key_instance_id, item_instance_cls in mask_info["instances"].items():
        if not isinstance(key_instance_id, str):
            raise ValueError(f"Instance ids need to be a str, found {type(key_instance_id)} "
                                f"of instance {key_instance_id} in {mask_info_path}")
        if not isinstance(item_instance_cls, int):
            raise ValueError(f"Instance classes needs to be an int, found {type(item_instance_cls)} "
                                f"of instance {key_instance_id} in {mask_info_path}")
        if not item_instance_cls in exp_labs:
            raise ValueError(f"Instance class {item_instance_cls} not defined in dataset.yml")


def _full_check(
    case_paths: List[Path],
    mask_info_path: Optional[Path] = None,
    ) -> None:
    """
    Performas itk and instance chekcs on provided paths

    Args:
        case_paths: paths to all itk images to check properties
            if label is provided it needs to be at the last position
        mask_info_path: optionally check label properties. If None, no
            check of label properties will be performed.

    Raises:
        ValueError: Inconsistent instances in label info and label image

    See also:
        :func:`_check_itk_params`
    """
    img_itk_seq = [load_sitk(cp) for cp in case_paths]
    _check_itk_params(img_itk_seq, case_paths)

    if mask_info_path is not None:
        mask_itk = img_itk_seq[-1]
        mask_info = load_json(mask_info_path)
        info_instances = list(map(int, mask_info["instances"].keys()))
        mask_instances = np.unique(sitk.GetArrayViewFromImage(mask_itk))
        mask_instances = mask_instances[mask_instances > 0]

        for mi in mask_instances:
            if not mi in info_instances:
                raise ValueError(f"Found instance ID {mi} in mask which is "
                                f"not present in info {info_instances} in {mask_info_path}")
        if not len(info_instances) == len(mask_instances):
            raise ValueError("Found instances in info which are not present in mask: "
                            f"mask: {mask_instances} info {info_instances} in {mask_info_path}")


def _check_itk_params(
    img_seq: Sequence[sitk.Image],
    paths: Sequence[Path],
) -> None:
    """
    Check Dimension, Origin, Direction and Spacing of a Sequence of images

    Args:
        img_seq: sequence of images to check
        paths: correcponding paths of images (for error msg)

    Raises:
        ValueError: raised if dimensions do not match
        ValueError: raised if origin does not match
        ValueError: raised if direction does not match
        ValueError: raised if spacing does not match
    """
    for idx, img in enumerate(img_seq[1:], start=1):
        if not (
            np.asarray(img_seq[0].GetDimension()) == np.asarray(img.GetDimension())
        ).all():
            raise ValueError(
                f"Expected {paths[idx]} and {paths[0]} to have same dimensions!"
            )
        if not ((np.asarray(img_seq[0].GetSize()) == np.asarray(img.GetSize()))).all():
            raise ValueError(
                f"Expected {paths[idx]} and {paths[0]} to have same dimensions!"
            )
        if not np.allclose(
            np.asarray(img_seq[0].GetOrigin()), np.asarray(img.GetOrigin())
        ):
            raise ValueError(
                f"Expected {paths[idx]} and {paths[0]} to have same origin!"
            )
        if not np.allclose(
            np.asarray(img_seq[0].GetDirection()), np.asarray(img.GetDirection())
        ):
            raise ValueError(
                f"Expected {paths[idx]} and {paths[0]} to have same direction!"
            )
        if not np.allclose(
            np.asarray(img_seq[0].GetSpacing()), np.asarray(img.GetSpacing())
        ):
            raise ValueError(
                f"Expected {paths[idx]} and {paths[0]} to have same spacing!"
            )
