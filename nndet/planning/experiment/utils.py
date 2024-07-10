# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import os
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from typing import Dict

import numpy as np
from loguru import logger

from nndet.io.dataformat import data_format_to_class_mapping
from nndet.io.itk import load_sitk_as_array
from nndet.io.load import load_json, load_pickle, save_pickle
from nndet.io.paths import get_case_ids_from_dir
from nndet.io.transforms.instances import (
    get_instance_class_from_properties_seq,
    instances_to_boxes_np,
    instances_to_segmentation_np,
)


def create_label_case(
    target_dir: Path,
    case_id: str,
    instances: np.ndarray,
    mapping: Dict[int, int],
    dim: int,
    properties: Dict,
) -> None:
    """
    Crete labels for evaluation and analysis purposes

    Args:
        target_dir: target dir to save labels
        case_id: case identifier
        instances: instance segmentation
        mapping: map each instance id to a class (classes start from 0)
        dim: spatial dimensions
        properties: pass through properties
    """
    instances_save_path = target_dir / f"{case_id}_instances_gt.npz"
    boxes_save_path = target_dir / f"{case_id}_boxes_gt.npz"
    seg_save_path = target_dir / f"{case_id}_seg_gt.npz"
    properties_save_path = target_dir / f"{case_id}.pkl"

    if instances_save_path.is_file() and boxes_save_path.is_file() and seg_save_path.is_file():
        logger.warning(f"Skipping prepare label {case_id} because it already exists")
    else:
        logger.info(f"Preparing label {case_id}")

        if instances.ndim == dim:
            instances = instances[None]
        assert instances.ndim == (dim + 1)

        np.savez_compressed(
            str(instances_save_path),
            instances=instances,
            mapping=mapping,
        )

        boxes, instance_idx = instances_to_boxes_np(seg=instances, dim=dim)
        box_classes = get_instance_class_from_properties_seq(instance_idx=instance_idx, map_dict=mapping)
        res = {"boxes": boxes, "classes": box_classes, "instance_idx": instance_idx}
        np.savez_compressed(str(boxes_save_path), **res)

        seg = instances_to_segmentation_np(instances, mapping)
        np.savez_compressed(str(seg_save_path), seg=seg)

        save_pickle(properties, properties_save_path)


def create_labels(
    preprocessed_output_dir: os.PathLike,
    source_dir: os.PathLike,
    num_processes: int = 6,
):
    """
    Creates labels for visualization and analysis purposes from raw labels
    Prepares: instance segmentation, bounding boxes, semantic segmentation

    Args:
        source_dir: base dir which containes labelsTr/labelsTs
        dim: number of spatial dimensions
        num_processes: number of processed to use
    """
    source_dir = Path(source_dir)
    for postfix in ["Tr", "Ts"]:
        if (source_label_dir := source_dir / f"labels{postfix}").is_dir():
            logger.info(f"Preparing {postfix} evaluation labels")
            target_dir = Path(preprocessed_output_dir) / f"labels{postfix}"
            target_dir.mkdir(parents=True, exist_ok=True)

            case_ids = get_case_ids_from_dir(
                source_label_dir,
                remove_modality=False,
                pattern="*.json",
            )
            if num_processes > 0:
                with Pool(processes=num_processes) as p:
                    p.starmap(
                        run_create_label,
                        zip(
                            repeat(source_label_dir),
                            case_ids,
                            repeat(3),
                            repeat(target_dir),
                        ),
                    )
            else:
                for cid in case_ids:
                    run_create_label(source_label_dir, cid, 3, target_dir)


def run_create_label(
    source_label_dir: Path,
    case_id: str,
    dim: int,
    target_dir: Path,
):
    """
    Helper to run preparation with multiprocessing

    Args:
        source_label_dir: directory with labels
        case_id: case id to process
        dim: number of spatial dimensions
        target_dir: directory to save results
    """
    instances = load_sitk_as_array(source_label_dir / f"{case_id}.nii.gz")[0]
    properties = load_json(source_label_dir / f"{case_id}.json")

    if instances.ndim == dim:
        instances = instances[None]
    instances = instances.astype(np.int32)

    mapping = {int(key): int(item) for key, item in properties["instances"].items()}

    create_label_case(
        target_dir=target_dir,
        case_id=case_id,
        instances=instances,
        mapping=mapping,
        dim=dim,
        properties=properties,
    )


def run_create_label_preprocessed(source_dir: Path, case_id: str, dim: int, target_dir: Path, data_format: str):
    """
    Helper to run preparation with multiprocessing

    Args:
        source_dir: directory with labels
        case_id: case id to process
        dim: number of spatial dimensions
        target_dir: directory to save results
        data_format: data format to save or load preprocessed data
    """
    with_data_format = data_format_to_class_mapping[data_format]
    instances = with_data_format.load_seg(source_dir / case_id)[:]
    properties = load_pickle(source_dir / f"{case_id}.pkl")

    mapping = {int(key): int(item) for key, item in properties["instances"].items()}

    create_label_case(
        target_dir=target_dir,
        case_id=case_id,
        instances=instances,
        mapping=mapping,
        dim=dim,
        properties=properties,
    )
