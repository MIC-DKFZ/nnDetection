import multiprocessing
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from itertools import repeat
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
from loguru import logger
from sklearn.model_selection import KFold, train_test_split

from nndet.io import load_sitk
from nndet.io.load import load_pickle, save_json, save_pickle
from nndet.io.paths import get_task
from nndet.utils.check import env_guard
from nndet.utils.info import maybe_verbose_iterable

POP_CASES_FROM_WEAK = [
    "sub-115",
    "sub-143",
    "sub-181",
    "sub-272",
]


def prepare_case(
    patient: Tuple[str, List[str]],
    source_data: Path,
    source_label_dir: Path,
    target_data_dir: Path,
    target_label_dir: Path,
) -> None:
    logger.info(f"Preparing patient {patient[0]}")
    for case_id in patient[1]:
        logger.info(f"Preparing case {case_id}")
        # folder structure: sub-XXX / ses-xxxxxx / anat / files
        patient_id, ses_id = case_id.split("_", 1)
        assert patient_id == patient[0], f"Patient id {patient_id} does not match {patient[0]}"
        data_dir = source_data / patient_id / ses_id / "anat"
        label_dir = source_label_dir / patient_id / ses_id / "anat"

        assert data_dir.is_dir(), f"Data dir {data_dir} does not exist"
        assert label_dir.is_dir(), f"Label dir {label_dir} does not exist"

        tof_file = data_dir / f"{patient_id}_{ses_id}_angio.nii.gz"
        assert tof_file.is_file(), f"TOF file {tof_file} does not exist"

        lesion_files = [
            p
            for p in label_dir.iterdir()
            if p.is_file() and p.name.startswith(f"{patient_id}_{ses_id}_desc-Lesion") and p.name.endswith(".nii.gz")
        ]
        lesion_files.sort()

        # handle data
        tof_itk = load_sitk(tof_file)
        sitk.WriteImage(tof_itk, str(target_data_dir / f"{case_id}_0000.nii.gz"))

        # handle label
        tof_np = sitk.GetArrayFromImage(tof_itk)
        lesion_mask_np = np.zeros_like(tof_np).astype(np.uint8)
        lesion_meta = {}

        if len(lesion_files) == 0:
            pass  # nothing todo no aneurysms
        else:
            for lesion_idx, lf in enumerate(lesion_files, start=1):
                lesion_itk = load_sitk(lf)
                lesion_np = sitk.GetArrayFromImage(lesion_itk)
                assert lesion_np.max() == 1, f"Lesion mask {lf} should be binary"
                assert lesion_np.min() == 0, f"Lesion mask {lf} should be binary"
                assert lesion_mask_np[lesion_np > 0].max() == 0, f"Lesion mask {lf} overlaps with another lesion"
                lesion_mask_np[lesion_np > 0] = lesion_idx
                lesion_meta[str(lesion_idx)] = 0

        lesion_mask_itk = sitk.GetImageFromArray(lesion_mask_np)
        lesion_mask_itk.CopyInformation(tof_itk)
        sitk.WriteImage(lesion_mask_itk, str(target_label_dir / f"{case_id}.nii.gz"))
        save_json({"instances": lesion_meta}, target_label_dir / f"{case_id}.json")


def custom_create_test_split(
    patient_dict: Dict[str, List[str]],  # all patients
    patient_subset: List[str],  # patient subset to actucally split
    splitted_dir: os.PathLike,
    num_modalities: int,
    test_size: float,
    random_state: int,
    shuffle: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Create custom test split
    """
    images_tr = Path(splitted_dir) / "imagesTr"
    labels_tr = Path(splitted_dir) / "labelsTr"
    images_ts = Path(splitted_dir) / "imagesTs"
    labels_ts = Path(splitted_dir) / "labelsTs"

    if not images_tr.is_dir():
        raise ValueError(f"No dir with training images found {images_tr}")
    if not labels_tr.is_dir():
        raise ValueError(f"No dir with training labels found {labels_tr}")
    images_ts.mkdir(parents=True, exist_ok=True)
    labels_ts.mkdir(parents=True, exist_ok=True)

    logger.info(f"Found {len(patient_subset)} to train/test split")
    train_patients, test_patients = train_test_split(
        patient_subset,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )
    logger.info(f"Using {train_patients} for training and {test_patients} for testing.")

    logger.info("Moving data ...")
    for pid in test_patients:
        for cid in patient_dict[pid]:
            for modality in range(num_modalities):
                shutil.move(
                    images_tr / f"{cid}_{modality:04d}.nii.gz",
                    images_ts / f"{cid}_{modality:04d}.nii.gz",
                )
            for p in labels_tr.glob(f"{cid}*"):
                shutil.move(p, labels_ts / p.name)
    logger.info("Finished moving data.")
    return train_patients, test_patients


def custom_create_cv_split(
    patient_dict: Dict[str, List[str]],
    voxel_level_patients: List[str],
    weak_level_patients: List[str],
    task: str,
    num_folds: int,
):
    """
    Create custom CV split
    """
    task_name = get_task(task, name=True)
    task_dir = Path(os.getenv("det_data")) / task_name
    preprocessed_dir = task_dir / "preprocessed"
    preprocessed_dir.mkdir(exist_ok=True)

    raw_splitted_dir = task_dir / "raw_splitted"
    if not raw_splitted_dir.is_dir():
        raise ValueError(f"{raw_splitted_dir} is not a directory!")
    label_dir = raw_splitted_dir / "labelsTr"
    if not label_dir.is_dir():
        raise ValueError(f"{label_dir} is not a directory!")

    splits_path_json = preprocessed_dir / "splits_final.json"
    splits_path_pkl = preprocessed_dir / "splits_final.pkl"

    if splits_path_json.is_file():
        raise ValueError(f"{splits_path_json} already exists.")
    if splits_path_pkl.is_file():
        raise ValueError(f"{splits_path_pkl} already exists.")

    # setup logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<level>{level}</level>: {message}",
        level="INFO",
        colorize=True,
    )
    logger.add(task_dir / "split.log", level="DEBUG")

    current_time = datetime.now()
    current_time_str = current_time.strftime("%d/%m/%Y %H:%M:%S")
    logger.info(f"+++ Running CUSTOM nndet_cv_split {current_time_str} +++")

    total_patients = len(voxel_level_patients) + len(weak_level_patients)
    logger.info(f"Found {total_patients} patients for CV split")

    # create stratified group k fold
    splits = []
    wl_cv = KFold(n_splits=num_folds, shuffle=True, random_state=0)
    vl_cv = KFold(n_splits=num_folds, shuffle=True, random_state=0)

    wl_cv_gen = wl_cv.split(weak_level_patients)  # patient ids
    vl_cv_gen = vl_cv.split(voxel_level_patients)  # patient ids

    for fold_idx in range(num_folds):
        wl_train_idx, wl_val_idx = next(wl_cv_gen)  # idx
        vl_train_idx, vl_val_idx = next(vl_cv_gen)  # idx

        train_pids = [weak_level_patients[_i] for _i in wl_train_idx] + [
            voxel_level_patients[_i] for _i in vl_train_idx
        ]
        val_pids = [weak_level_patients[_i] for _i in wl_val_idx] + [voxel_level_patients[_i] for _i in vl_val_idx]
        intersection_pids = set(train_pids).intersection(val_pids)
        assert not intersection_pids
        assert len(train_pids) + len(val_pids) == total_patients

        train_cids = [cid for pid in train_pids for cid in patient_dict[pid]]
        val_cids = [cid for pid in val_pids for cid in patient_dict[pid]]
        intersection_cids = set(train_cids).intersection(val_cids)
        assert not intersection_cids

        logger.info(
            f"Generated fold {fold_idx} with train:val :: "
            f"{len(train_pids)}:{len(val_pids)} patients "
            f"{len(train_cids)}:{len(val_cids)} cases "
        )
        splits.append({"train": train_cids, "val": val_cids})

    # save splits
    save_json(splits, splits_path_json)
    save_pickle(splits, splits_path_pkl)


@env_guard
def main():
    task = "Task052_MRAAneurysms"
    det_data_dir = Path(os.getenv("det_data"))
    task_data_dir = det_data_dir / task

    # setup raw paths
    source_data_dir = task_data_dir / "raw"
    source_label_dir = task_data_dir / "raw" / "derivatives" / "manual_masks"
    if not source_data_dir.is_dir():
        raise RuntimeError(f"{source_data_dir} should contain the raw data but does not exist.")
    if not source_label_dir.is_dir():
        raise RuntimeError(f"{source_label_dir} should contain the raw data but does not exist.")

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(task_data_dir / "prepare.log", level="DEBUG")

    # setup raw splitted dirs
    raw_splitted_dir = task_data_dir / "raw_splitted"
    target_data_dir = raw_splitted_dir / "imagesTr"
    target_data_dir.mkdir(exist_ok=True, parents=True)
    target_label_dir = raw_splitted_dir / "labelsTr"
    target_label_dir.mkdir(exist_ok=True, parents=True)

    # prepare dataset info
    meta = {
        "task": task,
        "dim": 3,
        "target_class": None,
        "test_labels": True,
        "labels": {"0": "aneurysm"},
        "modalities": {"0": "TOF"},
    }
    save_json(meta, task_data_dir / "dataset.json")

    # prepare data & label
    _patient_ids = [p.name for p in source_data_dir.iterdir() if p.is_dir() and p.name.startswith("sub-")]
    _patient_ids.sort()
    logger.info(f"Found {len(_patient_ids)} case ids")
    assert len(_patient_ids) == 284, "Missing cases"

    n_total = 0
    patient_dict: Dict[str, List[str]] = defaultdict(list)
    for _pid in _patient_ids:
        _sub_cases = [
            f"{_pid}_{p.name}" for p in (source_data_dir / _pid).iterdir() if p.is_dir() and p.name.startswith("ses-")
        ]
        patient_dict[_pid].extend(_sub_cases)
        n_total += len(_sub_cases)
    assert n_total == 296, "Missing images"
    assert len(patient_dict) == 284, "Missing cases"

    # parse information about annotation styles
    voxel_level_patients = load_pickle(
        source_data_dir / "Aneurysm_Detection" / "extra_files" / "patients_with_voxelwise_labels.pkl"
    )
    voxel_level_patients = list(set([v.rsplit("_", 1)[0] for v in voxel_level_patients]))
    voxel_level_patients.sort()
    weak_level_patients = load_pickle(
        source_data_dir / "Aneurysm_Detection" / "extra_files" / "patients_with_weak_labels.pkl"
    )
    weak_level_patients = [w.rsplit("_", 1)[0] for w in weak_level_patients]
    weak_level_patients = list(set([w for w in weak_level_patients if w not in POP_CASES_FROM_WEAK]))
    weak_level_patients.sort()
    assert len(voxel_level_patients) + len(weak_level_patients) == 284

    # create custom splits for this dataset
    if (raw_splitted_dir / "imagesTs").is_dir():
        raise ValueError(
            f"Test split already exists in {raw_splitted_dir}" f"Please remove it before running this script."
        )
    if (raw_splitted_dir / "labelsTs").is_dir():
        raise ValueError(
            f"Test split already exists in {raw_splitted_dir}" f"Please remove it before running this script."
        )
    if (task_data_dir / "preprocessed" / "splits_final.json").is_file():
        raise ValueError(f"CV split already exists in {task_data_dir}" f"Please remove it before running this script.")

    # start preparation
    num_processes = int(os.getenv("det_num_threads", 4))
    if num_processes > 0:
        # multiprocess version
        logger.info(f"Using {num_processes} processes for preparation")
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(
                prepare_case,
                zip(
                    patient_dict.items(),
                    repeat(source_data_dir),
                    repeat(source_label_dir),
                    repeat(target_data_dir),
                    repeat(target_label_dir),
                ),
            )
    else:
        # main process version
        logger.info("Using main process for preparation")
        for patient in maybe_verbose_iterable(patient_dict.items()):
            logger.info(f"Preparing patient: {patient[0]}")
            prepare_case(
                patient=patient,
                source_data=source_data_dir,
                source_label_dir=source_label_dir,
                target_data_dir=target_data_dir,
                target_label_dir=target_label_dir,
            )

    # create test split
    weak_train_patients, _ = custom_create_test_split(
        patient_dict=patient_dict,  # all cases
        patient_subset=weak_level_patients,  # patients with weak labels
        splitted_dir=raw_splitted_dir,
        num_modalities=len(meta["modalities"]),
        test_size=0.3,
        random_state=0,
        shuffle=True,
    )

    # create custom CV split
    custom_create_cv_split(
        patient_dict=patient_dict,  # all cases
        voxel_level_patients=voxel_level_patients,
        weak_level_patients=weak_train_patients,
        task=task,
        num_folds=5,
    )


if __name__ == "__main__":
    main()
