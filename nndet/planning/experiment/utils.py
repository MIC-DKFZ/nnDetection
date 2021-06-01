import os
from nndet.core.boxes.ops_np import box_size_np

import numpy as np
from pathlib import Path
from loguru import logger
from itertools import repeat
from multiprocessing import Pool
from typing import Dict

from nndet.io.itk import load_sitk_as_array
from nndet.io.load import load_json, load_pickle
from nndet.io.paths import get_case_ids_from_dir
from nndet.io.transforms.instances import (
    get_bbox_np,
    instances_to_segmentation_np,
    )


def create_label_case(
    target_dir: Path,
    case_id: str,
    instances: np.ndarray,
    mapping: Dict[int, int],
    dim: int,
    ) -> None:
    """
    Crete labels for evaluation and analysis purposes

    Args:
        target_dir: target dir to save labels
        case_id: case identifier
        instances: instance segmentation
        mapping: map each instance id to a class (classes start from 0)
        dim: spatial dimensions
    """
    instances_save_path = target_dir / f"{case_id}_instances_gt.npz"
    boxes_save_path = target_dir / f"{case_id}_boxes_gt.npz"
    seg_save_path = target_dir / f"{case_id}_seg_gt.npz"
    
    if instances_save_path.is_file() and boxes_save_path.is_file() and seg_save_path.is_file():
        logger.warning(f"Skipping prepare label {case_id} because it already exists")
    else:
        logger.info(f"Preparing label {case_id}")
        if instances.ndim == dim:
            instances = instances[None]
        np.savez_compressed(str(instances_save_path),
                            instances=instances, mapping=mapping,
                            )

        res = get_bbox_np(instances, mapping, dim=dim)
        np.savez_compressed(str(boxes_save_path), **res)

        seg = instances_to_segmentation_np(instances, mapping)
        np.savez_compressed(str(seg_save_path), seg=seg)


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
            logger.info(f'Preparing {postfix} evaluation labels')
            target_dir = Path(preprocessed_output_dir) / f"labels{postfix}"
            target_dir.mkdir(parents=True, exist_ok=True)

            case_ids = get_case_ids_from_dir(source_label_dir,
                                             remove_modality=False,
                                             pattern="*.json",
                                             )
            if num_processes > 0:
                with Pool(processes=num_processes) as p:
                    p.starmap(run_create_label,
                            zip(repeat(source_label_dir),
                                case_ids,
                                repeat(3),
                                repeat(target_dir),
                                )
                            )
            else:
                for cid in case_ids:
                    run_create_label(source_label_dir, cid, 3, target_dir)


def run_create_label(source_label_dir: Path,
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
    )


def run_create_label_preprocessed(
    source_dir: Path,
    case_id: str,
    dim: int,
    target_dir: Path,
    ):
    """
    Helper to run preparation with multiprocessing

    Args:
        source_dir: directory with labels
        case_id: case id to process
        dim: number of spatial dimensions
        target_dir: directory to save results
    """
    instances = np.load(str(source_dir / f"{case_id}.npz"), mmap_mode="r")["seg"]
    properties = load_pickle(source_dir / f"{case_id}.pkl")
    mapping = {int(key): int(item) for key, item in properties["instances"].items()}
    create_label_case(
        target_dir=target_dir,
        case_id=case_id,
        instances=instances,
        mapping=mapping,
        dim=dim,
    )
