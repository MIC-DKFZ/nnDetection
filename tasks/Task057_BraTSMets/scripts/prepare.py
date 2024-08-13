import concurrent.futures
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

import cc3d
import numpy as np
import pandas as pd
import scipy
import SimpleITK as sitk
from loguru import logger
from sklearn.model_selection import train_test_split

from nndet.io.itk import load_sitk
from nndet.io.load import save_json
from nndet.io.prepare import create_test_split
from nndet.utils.check import env_guard

DILATION_FACTOR = 1
MIN_VOLUME = 2  # mm^3

UCSF_EXCLUDE = [
    "100363B",
]


def convert_seg2mask(seg_itk: sitk.Image) -> Union[sitk.Image, Dict[int, int]]:
    seg = sitk.GetArrayFromImage(seg_itk)
    s0, s1, s2 = seg_itk.GetSpacing()
    volume_per_voxel = s0 * s1 * s2

    # process segmentation
    seg[seg == 1] = 1  # map NETC (nonenhancing tumor core)
    seg[seg == 2] = 0  # map SNFH (non-enhancing FLAIR hyperintensity)
    seg[seg == 3] = 1  # map ET (enhancing tumor)

    seg_cc = cc3d.connected_components(seg, connectivity=26)

    dilation_struct = scipy.ndimage.generate_binary_structure(3, 2)
    seg_dilated = scipy.ndimage.binary_dilation(seg, structure=dilation_struct, iterations=DILATION_FACTOR)
    seg_dilated_cc = cc3d.connected_components(seg_dilated, connectivity=26)

    # create mask
    mask = np.zeros_like(seg)
    instances = {}
    instance_idx = 1
    skipped_idx = 0
    for dilated_instance_id in range(1, np.max(seg_dilated_cc) + 1):
        dilated_instance_mask = (seg_dilated_cc == dilated_instance_id).astype(int)
        instance_mask = ((seg_cc * dilated_instance_mask) > 0).astype(bool)  # filter instance

        if instance_mask.sum() * volume_per_voxel < MIN_VOLUME:
            skipped_idx = skipped_idx + 1
        else:
            mask[instance_mask] = instance_idx
            instances[instance_idx] = 0
            instance_idx = instance_idx + 1

    assert mask.max() == seg_dilated_cc.max() - skipped_idx, f"{mask.max()} != {seg_dilated_cc.max()}"

    # create mask itk
    mask_itk = sitk.GetImageFromArray(mask)
    mask_itk.CopyInformation(seg_itk)
    return mask_itk, instances


def new_brats_case_id(case_id: str) -> str:
    patient_id = "BRATS-" + case_id.rsplit("-", 1)[0].replace("-", "")
    session_id = case_id.rsplit("-", 1)[1]
    return f"{patient_id}_{session_id}"


def new_ucsf_case_id(case_id: str) -> str:
    patient_id = "UCFS-" + case_id[:-1]
    session_id = ord(case_id[-1:].lower()) - ord("a")
    return f"{patient_id}_{session_id:03d}"


def prepare_case(args) -> None:
    case_id, new_case_id, source_data, target_data_dir, target_label_dir = args
    logger.info(f"Preparing case {case_id} -> {new_case_id}")
    if new_case_id.startswith("BRATS"):
        prepare_case_brats(
            case_id=case_id,
            new_case_id=new_case_id,
            source_data=source_data,
            target_data_dir=target_data_dir,
            target_label_dir=target_label_dir,
        )
    elif new_case_id.startswith("UCFS"):
        prepare_case_ucsf(
            case_id=case_id,
            new_case_id=new_case_id,
            source_data=source_data,
            target_data_dir=target_data_dir,
            target_label_dir=target_label_dir,
        )
    else:
        raise ValueError(f"Unknown case id {new_case_id}")


def prepare_case_brats(
    case_id: str,
    new_case_id: str,
    source_data: Path,
    target_data_dir: Path,
    target_label_dir: Path,
) -> None:
    # copy data
    shutil.copy2(
        source_data / case_id / f"{case_id}-t1n.nii.gz",
        target_data_dir / f"{new_case_id}_0000.nii.gz",
    )
    shutil.copy2(
        source_data / case_id / f"{case_id}-t1c.nii.gz",
        target_data_dir / f"{new_case_id}_0001.nii.gz",
    )
    shutil.copy2(
        source_data / case_id / f"{case_id}-t2f.nii.gz",
        target_data_dir / f"{new_case_id}_0002.nii.gz",
    )
    # shutil.copy2(source_data / case_id / f"{case_id}-t2w.nii.gz", target_data_dir / f"{case_id}_000_0003.nii.gz")

    # copy seg
    seg_itk = load_sitk(source_data / case_id / f"{case_id}-seg.nii.gz")
    mask_itk, instances = convert_seg2mask(seg_itk)
    sitk.WriteImage(mask_itk, target_label_dir / f"{new_case_id}.nii.gz")
    save_json({"instances": instances, "orig_case_id": case_id}, target_label_dir / f"{new_case_id}.json")


def prepare_case_ucsf(
    case_id: str,
    new_case_id: str,
    source_data: Path,
    target_data_dir: Path,
    target_label_dir: Path,
) -> None:
    # copy data
    shutil.copy2(
        source_data / case_id / f"{case_id}_T1pre.nii.gz",
        target_data_dir / f"{new_case_id}_0000.nii.gz",
    )
    shutil.copy2(
        source_data / case_id / f"{case_id}_T1post.nii.gz",
        target_data_dir / f"{new_case_id}_0001.nii.gz",
    )
    shutil.copy2(
        source_data / case_id / f"{case_id}_FLAIR.nii.gz",
        target_data_dir / f"{new_case_id}_0002.nii.gz",
    )

    # copy seg
    seg_itk = load_sitk(source_data / case_id / f"{case_id}_BRaTS-seg.nii.gz")
    mask_itk, instances = convert_seg2mask(seg_itk)
    sitk.WriteImage(mask_itk, target_label_dir / f"{new_case_id}.nii.gz")
    save_json({"instances": instances, "orig_case_id": case_id}, target_label_dir / f"{new_case_id}.json")


def create_patient_mapper(case_ids: List[str]) -> Dict[str, List[str]]:
    patient_mapper = defaultdict(list)
    for case_id in case_ids:
        patient_id = case_id.split("_")[0]
        patient_mapper[patient_id].append(case_id)
    return patient_mapper


def create_test_split_custom(
    patient_mapper: Dict[str, List[str]],
    splitted_dir: os.PathLike,
    num_modalities: int,
    test_size: float = 0.3,
    random_state: int = 0,
    shuffle: bool = True,
):
    """
    Helper function to create an artificial test split from the splitted data
    Performs splitting on patient level

    Args:
        patient_mapper: map patient ids to case ids
        splitted_dir: path to directory with splitted data. `imagesTr` and
            `labelsTr` need to exist beforehand. `imagesTs` and `labelsTs`
            will be created automatically.
        num_modalities: number of modalities
        test_size: size of test set, needs to be a value between 0 and 1
        random_state: seed for splitting
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

    pids = list(patient_mapper.keys())
    train_pids, test_pids = train_test_split(
        pids,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=None,
    )
    logger.info(f"Using {train_pids} for training and {test_pids} for testing.")

    logger.info("Moving data ...")
    for pid in test_pids:
        for cid in patient_mapper[pid]:
            for modality in range(num_modalities):
                shutil.move(
                    images_tr / f"{cid}_{modality:04d}.nii.gz",
                    images_ts / f"{cid}_{modality:04d}.nii.gz",
                )
            for p in labels_tr.glob(f"{cid}*"):
                shutil.move(p, labels_ts / p.name)


@env_guard
def main():
    task = "Task057_BraTSMets"
    det_data_dir = Path(os.getenv("det_data"))
    task_data_dir = det_data_dir / task

    # setup raw paths
    source_data_brats_base = task_data_dir / "raw" / "ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData"
    if not source_data_brats_base.is_dir():
        raise RuntimeError(f"{source_data_brats_base} should contain the raw data but does not exist.")
    source_data_brats_add = task_data_dir / "raw" / "ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData_Additional"
    if not source_data_brats_add.is_dir():
        raise RuntimeError(f"{source_data_brats_add} should contain the raw data but does not exist.")
    source_data_ucsf = task_data_dir / "raw" / "UCSF_BrainMetastases_v1.3" / "UCSF_BrainMetastases_TRAIN"
    if not source_data_ucsf.is_dir():
        raise RuntimeError(f"{source_data_ucsf} should contain the raw data but does not exist.")
    source_ucsf_file = (
        task_data_dir / "raw" / "UCSF_BrainMetastases_v1.3" / "TableS1_UCSF_BrainMetastases_SubjectInfo.xlsx"
    )
    if not source_ucsf_file.is_file():
        raise RuntimeError(f"{source_ucsf_file} should contain the raw data but does not exist.")

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
        "target_class": None,
        "test_labels": True,
        "labels": {"0": "tumor"},
        "modalities": {"0": "T1n", "1": "T1c", "2": "FLAIR"},
        "dim": 3,
        # Multiple scans per patient available
        "session_id": True,
    }
    save_json(meta, task_data_dir / "dataset.json")

    # parse case ids and prepare new case ids
    base_case_ids = [p.name for p in source_data_brats_base.iterdir() if p.is_dir()]
    add_case_ids = [p.name for p in source_data_brats_add.iterdir() if p.is_dir()]
    df_ucsf = pd.read_excel(source_ucsf_file)
    df_ucsf = df_ucsf.dropna(subset=["BraTS_ID"])
    ucsf_case_ids = list(df_ucsf["SubjectID"])
    ucsf_case_ids = [cid for cid in ucsf_case_ids if cid not in UCSF_EXCLUDE]

    assert len(base_case_ids) == 165
    assert len(add_case_ids) == 73
    assert len(ucsf_case_ids) == (324 - len(UCSF_EXCLUDE))

    base_case_ids.sort()
    add_case_ids.sort()
    ucsf_case_ids.sort()

    logger.info(f"Found {len(base_case_ids)} base case ids")
    logger.info(f"Found {len(add_case_ids)} additional case ids")
    logger.info(f"Found {len(ucsf_case_ids)} UCSF case ids")

    new_base_case_ids = [new_brats_case_id(cid) for cid in base_case_ids]
    new_add_case_ids = [new_brats_case_id(cid) for cid in add_case_ids]
    new_ucsf_case_ids = [new_ucsf_case_id(cid) for cid in ucsf_case_ids]

    # prepare cases
    prepare_mapper = []
    for cid, new_cid in zip(base_case_ids, new_base_case_ids):
        prepare_mapper.append((cid, new_cid, source_data_brats_base, target_data_dir, target_label_dir))
    for cid, new_cid in zip(add_case_ids, new_add_case_ids):
        prepare_mapper.append((cid, new_cid, source_data_brats_add, target_data_dir, target_label_dir))
    for cid, new_cid in zip(ucsf_case_ids, new_ucsf_case_ids):
        prepare_mapper.append((cid, new_cid, source_data_ucsf, target_data_dir, target_label_dir))

    num_processes = int(os.getenv("det_num_threads", 4))
    if num_processes < 1:
        logger.info("Running in single process mode")
        for args in prepare_mapper:
            prepare_case(args)
    else:
        logger.info(f"Running in multi process mode with {num_processes} processes")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            for _ in executor.map(
                prepare_case,
                prepare_mapper,
            ):
                pass

    ################## split ##################
    # create a custo test split
    # brats and ucsf are split separately
    brats_case_ids_new = new_base_case_ids + new_add_case_ids
    brats_patient_mapper = create_patient_mapper(brats_case_ids_new)
    # brats has unique patients
    for key, value in brats_patient_mapper.items():
        assert len(value) == 1

    create_test_split(
        splitted_dir=raw_splitted_dir,
        num_modalities=len(meta["modalities"]),
        test_size=0.3,
        random_state=0,
        shuffle=True,
        do_stratify=True,
        case_ids=brats_case_ids_new,
    )

    # ucsf has multiple scans per patient -> need to split by patient
    ucsf_patient_mapper = create_patient_mapper(new_ucsf_case_ids)
    create_test_split_custom(
        patient_mapper=ucsf_patient_mapper,
        splitted_dir=raw_splitted_dir,
        num_modalities=len(meta["modalities"]),
        test_size=0.3,
        random_state=0,
        shuffle=True,
    )


if __name__ == "__main__":
    main()
