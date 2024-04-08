import argparse
import os
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pylidc as pl
import SimpleITK as sitk
from loguru import logger

from nndet.io.load import load_json, save_json
from nndet.utils.check import env_guard

MAX_ANNOTATORS = 4
MALIGNANCY_THRESHOLD = 3

# automatic clustering failed
MANUAL_CLUSTERING_INT = [
    55,
    92,
    137,
    204,
    252,
    332,
    340,
    366,
    404,
    608,
    815,
    863,
    865,
    942,
]
MANUAL_CLUSTERING_LIDC = [f"LIDC-IDRI-{i:04d}" for i in MANUAL_CLUSTERING_INT]
MANUAL_CLUSTER_LIDC_IMG_IDX = {"LIDC-IDRI-0332": 1}

# sitk load error
DIFFERENT_LOAD_INT = [85, 146, 418, 572, 979]
DIFFERENT_LOAD_LIDC = [f"LIDC-IDRI-{i:04d}" for i in DIFFERENT_LOAD_INT]

# some int cases are missing
MISSING_CASES_INT = [238, 585]


def get_manual_clustering(case_identifier: str, img_idx: int, scan_id: int) -> List[List[int]]:
    manual_grps = load_json(Path(__file__).parent / "group.json")
    int_id = int(case_identifier.split("-")[-1])

    case_grp = manual_grps[str(int_id)]
    assert img_idx == case_grp.get("img_idx", 0), f"img_idx: {img_idx} != {case_grp.get('img_idx', 0)}"
    assert scan_id == int(case_grp["scan_id"]), f"scan_id: {scan_id} != {case_grp['scan_id']}"
    return case_grp["groups"]


def group_annotations(groups: List[List[int]], annotations):
    ann_dict = {an.id: an for an in annotations}
    return [[ann_dict[i] for i in grp] for grp in groups]


class ShapeMismatchError(Exception):
    pass


def read_dcm_sitk(path_to_dcm: str) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path_to_dcm)
    reader.SetFileNames(dicom_names)
    img_itk = reader.Execute()
    return img_itk


def create_mask(
    mask_shape: Sequence[int],
    annotations: Sequence[pl.Annotation],
    min_lesion_votes: str,
    mask_voting: str,
    class_vote: str,
) -> np.ndarray:
    """
    min_lesion_votes: minimum number of annotators to label the lesion
        (inclsuive)

    Mask Voting can be one of:
        `union`: at least one annotator marked the voxel
        `lesion_majority`: majority voting per lesion (only annotators for
            the respective lesion are counted)
        `annotator_majority`: majority voting per voxel (all annotators are
            counted)

    class_vote can be one of:
        `none`: all lesions have class 0
        `old`: no annotations are counted as zeros, if mean malignancy is
            is equal or greater than 3, the lesion is considered malignant
        `lesion_majority`: majority voted per lesion (only annotators for
            the respective lesion are included)
    """
    mask_np = np.zeros(mask_shape, dtype=np.int32)

    lesion_index = 0  # iterate number of lesions
    lesion_meta = {
        "instances": {},
        "orig_malignancy": defaultdict(list),
        "orig_texture": defaultdict(list),
    }
    for lesion_anns in annotations:  # iterate lesions
        if len(lesion_anns) >= min_lesion_votes:
            lesion_index += 1
            # build temporary mask
            tmp_mask = np.zeros_like(mask_np)
            for annotator_lesion_ann in lesion_anns:
                tmp_mask[annotator_lesion_ann.bbox()][annotator_lesion_ann.boolean_mask()] += 1

            if mask_voting == "union":
                tmp_mask_bin = tmp_mask > 0
            elif mask_voting == "lesion_majority":
                tmp_mask_bin = tmp_mask >= len(lesion_anns) / 2
            elif mask_voting == "lesion_majority_abs":
                tmp_mask_bin = tmp_mask > len(lesion_anns) / 2
            elif mask_voting == "annotator_majority":
                tmp_mask_bin = tmp_mask >= MAX_ANNOTATORS / 2
            else:
                raise ValueError(f"Unknown mask_voting: {mask_voting}")

            if tmp_mask_bin.max() == 0:
                # sometimes disjunct masks are grouped together without overlap (2 instances in the entire dataset..)
                # we need to roll back our index and continue in these cases since the mask would be empty
                lesion_index -= 1
                continue
            else:
                # assign values for mask
                for annotator_lesion_ann in lesion_anns:
                    lesion_meta["orig_malignancy"][lesion_index].append(annotator_lesion_ann.malignancy)
                    lesion_meta["orig_texture"][lesion_index].append(annotator_lesion_ann.texture)
                mask_np[tmp_mask_bin] = lesion_index

    for lesion_idx, lesion_malignancy in lesion_meta["orig_malignancy"].items():
        if class_vote == "none":
            lesion_meta["instances"][lesion_idx] = 0
        elif class_vote == "old":
            _t = lesion_malignancy + [0] * (MAX_ANNOTATORS - len(lesion_malignancy))
            lesion_meta["instances"][lesion_idx] = int(np.mean(_t) >= MALIGNANCY_THRESHOLD)
        elif class_vote == "lesion_majority":
            counter = Counter(lesion_malignancy)
            lesion_meta["instances"][lesion_idx] = counter.most_common(1)[0][0]
        else:
            raise ValueError(f"Unknown class_vote: {class_vote}")

    return mask_np, lesion_meta


def prepare_case_union_bin(
    case_identifier: str,
    min_lesion_votes: int = 1,
    mask_voting="union",
    class_vote="none",
    perform_size_check: bool = True,
) -> List[Tuple[sitk.Image, sitk.Image, dict]]:
    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == case_identifier)

    patient_data = []
    for scan_idx, scan in enumerate(scans):
        matched_case_identifier = case_identifier in MANUAL_CLUSTERING_LIDC
        matched_img_idx = scan_idx == MANUAL_CLUSTER_LIDC_IMG_IDX.get(case_identifier, 0)
        if matched_case_identifier and matched_img_idx:
            logger.info(f"Using manual clustering for {case_identifier}")
            annotation_ids = get_manual_clustering(case_identifier, scan_idx, scan.id)
            annotations = group_annotations(annotation_ids, scan.annotations)
        else:
            annotations = scan.cluster_annotations()

        img_itk = read_dcm_sitk(scan.get_path_to_dicom_files())
        if case_identifier in DIFFERENT_LOAD_LIDC:
            logger.info(f"Using differnet loading for {case_identifier}")
            scan_vol = scan.to_volume()  # x, y, z
            mask_shape = scan_vol.shape

            orig_img_itk = img_itk
            img_np = scan_vol.transpose(2, 0, 1)  # z, y, x
            img_itk = sitk.GetImageFromArray(img_np)
            img_itk.SetOrigin(orig_img_itk.GetOrigin())
            img_itk.SetSpacing(orig_img_itk.GetSpacing())
            img_itk.SetDirection(orig_img_itk.GetDirection())
        elif perform_size_check:
            mask_shape = img_itk.GetSize()  # x, y, z
            scan_shape = scan.to_volume().shape
            if not mask_shape == scan_shape:
                logger.error(f"Case {case_identifier} has different size. Itk image {mask_shape} vs. scan {scan_shape}")
                raise ShapeMismatchError()
        else:
            mask_shape = img_itk.GetSize()  # x, y, z

        mask_np, mask_meta = create_mask(
            mask_shape=mask_shape,  # x, y, z
            annotations=annotations,  # x, y, z
            min_lesion_votes=min_lesion_votes,
            mask_voting=mask_voting,
            class_vote=class_vote,
        )
        assert mask_np.ndim == 3
        mask_np = mask_np.transpose(2, 0, 1)  # z, y, x

        mask_itk = sitk.GetImageFromArray(mask_np)
        mask_itk.CopyInformation(img_itk)

        patient_data.append((img_itk, mask_itk, mask_meta))
    return patient_data


@env_guard
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare LIDC dataset for nnDetection with pylidc. " "If malignant is active, Task045 will be created."
        )
    )
    parser.add_argument(
        "--malignant",
        action="store_true",
        help="Classes will be splitted into benign and malignant",
    )
    args = parser.parse_args()
    malignant = args.malignant

    if malignant:
        t = "Task045_LIDC_pylidc_malignant"
        target_class = 1
        labels = {"0": "benign", "1": "malignant"}
        class_vote = "old"
    else:
        t = "Task044_LIDC_pylidc"
        target_class = 0
        labels = {"0": "nodule"}
        class_vote = "none"

    det_data_dir = Path(os.getenv("det_data"))
    task_data_dir = det_data_dir / t

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(task_data_dir / "prepare.log", level="DEBUG")

    # setup raw splitted dirs
    target_data_dir = task_data_dir / "raw_splitted" / "imagesTr"
    target_data_dir.mkdir(exist_ok=True, parents=True)
    target_label_dir = task_data_dir / "raw_splitted" / "labelsTr"
    target_label_dir.mkdir(exist_ok=True, parents=True)

    # prepare dataset info
    meta = {
        "task": t,
        "target_class": target_class,
        "test_labels": False,
        "labels": labels,
        "modalities": {"0": "CT"},
        "dim": 3,
    }
    save_json(meta, task_data_dir / "dataset.json")

    for idx in range(1, 1013):
        if idx in MISSING_CASES_INT:
            logger.info(f"Skipping missing case {idx}")
            continue

        case_identifier = f"LIDC-IDRI-{idx:04d}"
        logger.info(f"Processing {case_identifier}")
        try:
            patient_data = prepare_case_union_bin(
                case_identifier,
                min_lesion_votes=2,
                mask_voting="lesion_majority_abs",
                class_vote=class_vote,
            )
            for i, (img_itk, mask_itk, mask_meta) in enumerate(patient_data):
                logger.info(f"Writing {case_identifier} with img idx {i}")
                sitk.WriteImage(img_itk, target_data_dir / f"lidc{idx:04d}_{i:03d}_0000.nii.gz")
                sitk.WriteImage(mask_itk, target_label_dir / f"lidc{idx:04d}_{i:03d}.nii.gz")
                save_json(mask_meta, target_label_dir / f"lidc{idx:04d}_{i:03d}.json")
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(str(e))
        logger.info(f"Finished {case_identifier}")


if __name__ == "__main__":
    main()
