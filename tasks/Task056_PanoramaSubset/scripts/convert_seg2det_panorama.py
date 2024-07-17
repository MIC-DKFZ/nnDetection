# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
import traceback
from datetime import datetime
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from typing import Sequence

import numpy as np
import SimpleITK as sitk
from hydra import initialize_config_module
from loguru import logger
from tqdm import tqdm

import nndet.core.ops_np as ops_np
from nndet.io import save_json
from nndet.io.itk import load_sitk, load_sitk_as_array
from nndet.io.load import load_json
from nndet.io.transforms.instances import instances_to_boxes_np
from nndet.utils.check import env_guard
from nndet.utils.clustering import seg_to_instances
from nndet.utils.config import compose

"""
Modififed from nndetection script to append additional information to the json file
Rest is the same
Everything >6 will be set to background in the semantic segmentation
"""


def prepare_detection_label(
    case_id: str,
    label_dir: Path,
    things_classes: Sequence[int],
    stuff_classes: Sequence[int],
    min_size: float = 0,  # voxel
    min_vol: float = 0,  # voxel
    min_vol_mm3: float = 0,  # mm^3
):
    try:
        logger.info(f"Processing {case_id}")
        seg_itk = load_sitk(label_dir / f"{case_id}.nii.gz")
        spacing = np.asarray(seg_itk.GetSpacing())[::-1]
        seg = sitk.GetArrayFromImage(seg_itk)

        seg[seg > 6] = 0

        if len(seg_itk.GetSize()) == 3:
            voxel_mm3 = seg_itk.GetSpacing()[0] * seg_itk.GetSpacing()[1] * seg_itk.GetSpacing()[2]
        elif seg_itk.GetSize() == 2:
            voxel_mm3 = seg_itk.GetSpacing()[0] * seg_itk.GetSpacing()[1]
        else:
            raise ValueError(f"Unknown dimension {seg_itk.GetSize()} expected to have 2 or 3 dimensions")

        # prepare stuff information
        stuff_seg = np.zeros_like(seg)
        if stuff_classes:
            for new_class, old_class in enumerate(stuff_classes, start=1):
                stuff_seg[seg == old_class] = new_class
            stuff_seg_itk = sitk.GetImageFromArray(stuff_seg)
            stuff_seg_itk.SetOrigin(seg_itk.GetOrigin())
            stuff_seg_itk.SetDirection(seg_itk.GetDirection())
            stuff_seg_itk.SetSpacing(seg_itk.GetSpacing())

            sitk.WriteImage(stuff_seg_itk, str(label_dir / f"{case_id}_stuff.nii.gz"))

        # prepare things information
        things_seg = np.copy(seg)
        things_seg[stuff_seg > 0] = 0  # remove all stuff classes from segmentation

        instances_not_filtered, instances_not_filtered_classes = seg_to_instances(things_seg)
        final_mapping = {}
        if instances_not_filtered.max() > 0:
            boxes, instance_ids = instances_to_boxes_np(seg=instances_not_filtered, dim=instances_not_filtered.ndim)
            box_sizes = ops_np.box_size_np(boxes)

            assert len(instance_ids) == len(boxes)
            isotopic_axis = list(range(seg.ndim))
            isotopic_axis.pop(np.argmax(spacing))
            instances = np.zeros_like(instances_not_filtered)

            start_id = 1
            for iid, bsize in zip(instance_ids, box_sizes):
                bsize_world = bsize * spacing
                instance_mask = instances_not_filtered == iid
                instance_vol = instance_mask.sum()
                instance_vol_mm3 = instance_vol * voxel_mm3

                if (
                    all(bsize_world[isotopic_axis] > min_size)
                    and (instance_vol > min_vol)
                    and (instance_vol_mm3 > min_vol_mm3)
                ):
                    instances[instance_mask] = start_id
                    semantic_class = instances_not_filtered_classes[int(iid)]
                    final_mapping[start_id] = things_classes.index(semantic_class)
                    start_id += 1
        else:
            instances = np.zeros_like(instances_not_filtered)

        final_instances_itk = sitk.GetImageFromArray(instances)
        final_instances_itk.SetOrigin(seg_itk.GetOrigin())
        final_instances_itk.SetDirection(seg_itk.GetDirection())
        final_instances_itk.SetSpacing(seg_itk.GetSpacing())

        sitk.WriteImage(final_instances_itk, str(label_dir / f"{case_id}.nii.gz"))

        current_json = load_json(label_dir / f"{case_id}.json")
        current_json["instances"] = final_mapping
        save_json(current_json, label_dir / f"{case_id}.json")

        sitk.WriteImage(seg_itk, str(label_dir / f"{case_id}_orig.nii.gz"))
    except Exception as e:
        logger.error(f"Failed to process {case_id} with error: \n{e} and traceback: \n{traceback.format_exc()}")


@env_guard
def main():
    """
    This script converts a semantic segmentation dataset into an instance
    segmentation dataset by using connected components on the labels.
    To account for separated pixels inside the annotations only annotations
    with a specified minimal size are converted into objects.

    The data needs to be in the same format as in nnunet: images
    stay the same, labels will be semantic segmentations.

    ============================================================================
    ================================IMPORTANT==================================+
    ============================================================================
    Needs additional information from dataset.json/.yaml:
        `seg2det_stuff`: these are classes which are interpreted semantically
            (stuff classes are experimental and will probably changed in
             the future)
        `seg2det_things`: these are classes which are interpreted as instances
         Both entries should be lists with the indices of the respective
         classes where the position will determine its new class
         (currently only one classes is supported here)
         e.g.
            `seg2det_stuff`: [2,] -> remap class 2 from semantic segmentation
                to new stuff class 1 (stuff classes start at one)
            `seg2det_things`: [1, 3] -> remap class 1 and 3 from semantic
                segmentation to new things classes 0 and 1, respectively
            `min_size`: minimum size in mm of objects in the isotropic axis (default 0)
            `min_vol`: minimum volume of instances in pixels (default 0)
    ============================================================================

    The segmentation labels will be splitted into things (classes to detect)
    and  stuff classes (additional segmentation labels) and will be saved
    as separate files.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tasks",
        type=str,
        nargs="+",
        help="Single or multiple task identifiers to process consecutively",
    )
    parser.add_argument(
        "-o",
        "--overwrites",
        type=str,
        nargs="+",
        help="overwrites for config file",
        required=False,
    )
    parser.add_argument(
        "--volume_ranking",
        help="Create a ranking of instances based on their volume",
        action="store_true",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        required=False,
        help="Number of processes to use for conversion. Default 4.",
    )

    args = parser.parse_args()
    tasks = args.tasks
    ov = args.overwrites
    do_volume_ranking = args.volume_ranking
    num_processes = args.num_processes
    initialize_config_module(config_module="nndet.conf", version_base="1.1")

    for task in tasks:
        cfg = compose(task, "config.yaml", overrides=ov if ov is not None else [])
        print(cfg)

        splitted_dir = Path(os.getenv("det_data")) / cfg["task"] / "raw_splitted"

        logger.remove()
        logger.add(sys.stdout, level="INFO")
        logger.add(splitted_dir / "convert_seg2det.log", level="DEBUG")
        logger.info(f"+++++ Running covnersion: {datetime.now()} +++++")

        _seg2det_things = list(cfg["data"]["seg2det_things"])
        _seg2det_stuff = list(cfg["data"]["seg2det_stuff"])
        _min_size = cfg["data"].get("min_size", 0)
        _min_vol = cfg["data"].get("min_vol", 0)
        _min_vol_mm3 = cfg["data"].get("min_vol_mm3", 0)

        logger.info(f"Running conversion with seg2det_things {_seg2det_things} and seg2det_stuff {_seg2det_stuff}")
        logger.info(f"Running min_size {_min_size} and min_vol {_min_vol} and min_vol_mm3 {_min_vol_mm3}")

        for postfix in ["Tr", "Ts"]:
            label_dir = splitted_dir / f"labels{postfix}"
            case_ids = [f.name[:-7] for f in label_dir.glob("*.nii.gz")]
            case_ids.sort()
            logger.info(f"Found {len(case_ids)} cases for conversion with postfix {postfix}.")

            if num_processes == 0:
                for cid in case_ids:
                    prepare_detection_label(
                        case_id=cid,
                        label_dir=label_dir,
                        things_classes=_seg2det_things,
                        stuff_classes=_seg2det_stuff,
                        min_size=_min_size,
                        min_vol=_min_vol,
                        min_vol_mm3=_min_vol_mm3,
                    )
            else:
                with Pool(processes=num_processes) as p:
                    p.starmap(
                        prepare_detection_label,
                        zip(
                            case_ids,
                            repeat(label_dir),
                            repeat(_seg2det_things),
                            repeat(_seg2det_stuff),
                            repeat(_min_size),
                            repeat(_min_vol),
                            repeat(_min_vol_mm3),
                        ),
                    )

        if do_volume_ranking:
            for postfix in ["Tr", "Ts"]:
                if (label_dir := splitted_dir / f"labels{postfix}").is_dir():
                    ranking = []
                    for case_id in tqdm([f.stem for f in label_dir.glob("*.json")]):
                        instances = load_sitk_as_array(label_dir / f"{case_id}.nii.gz")[0]
                        instance_ids, instance_counts = np.unique(instances, return_counts=True)
                        cps = [np.argwhere(instances == iid)[0].tolist() for iid in instance_ids[1:]]
                        assert len(instance_ids) - 1 == len(cps)
                        tmp = [
                            {
                                "case_id": str(case_id),
                                "instance_id": int(iid),
                                "vol": int(vol),
                                "cp": list(cp)[::-1],
                            }
                            for iid, vol, cp in zip(instance_ids[1:], instance_counts[1:], cps)
                        ]
                        ranking.extend(tmp)
                    ranking = sorted(ranking, key=lambda x: x["vol"])
                    save_json(ranking, splitted_dir / f"volume_ranking_{postfix}.json")
                else:
                    logger.info(f"Did not find dir {label_dir} for volume ranking")


if __name__ == "__main__":
    main()
