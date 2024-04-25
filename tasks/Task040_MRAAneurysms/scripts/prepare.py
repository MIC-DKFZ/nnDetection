import multiprocessing
import os
import sys
from itertools import repeat
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from loguru import logger

from nndet.io import load_sitk
from nndet.io.load import save_json
from nndet.io.prepare import create_test_split
from nndet.utils.check import env_guard
from nndet.utils.info import maybe_verbose_iterable


def prepare_case(
    case_id: str,
    source_data: Path,
    source_label_dir: Path,
    target_data_dir: Path,
    target_label_dir: Path,
) -> None:
    logger.info(f"Preparing case {case_id}")

    # folder structure: sub-XXX / ses-xxxxxx / anat / files
    data_sub = [p for p in (source_data / case_id).iterdir() if p.is_dir() and p.name.startswith("ses-")]
    assert len(data_sub)
    session_name = data_sub[0].name
    data_dir = source_data / case_id / session_name / "anat"
    label_dir = source_label_dir / case_id / session_name / "anat"

    assert data_dir.is_dir(), f"Data dir {data_dir} does not exist"
    assert label_dir.is_dir(), f"Label dir {label_dir} does not exist"

    # t1_file = data_dir / f"{case_id}_{session_name}_T1w.nii.gz"
    tof_file = data_dir / f"{case_id}_{session_name}_angio.nii.gz"

    # assert t1_file.is_file(), f"T1 file {t1_file} does not exist"
    assert tof_file.is_file(), f"TOF file {tof_file} does not exist"

    lesion_files = [
        p
        for p in label_dir.iterdir()
        if p.is_file() and p.name.startswith(f"{case_id}_{session_name}_desc-Lesion") and p.name.endswith(".nii.gz")
    ]
    lesion_files.sort()

    # handle data
    tof_itk = load_sitk(tof_file)
    sitk.WriteImage(tof_itk, str(target_data_dir / f"{case_id}_0000.nii.gz"))

    # t1_itk = load_sitk(t1_file)
    # resampler = sitk.ResampleImageFilter()
    # resampler.SetInterpolator(sitk.sitkBSpline)
    # resampler.SetReferenceImage(tof_itk)
    # t1_resampled_itk = resampler.Execute(t1_itk)
    # sitk.WriteImage(t1_resampled_itk, str(target_data_dir / f"{case_id}_0000.nii.gz"))

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


@env_guard
def main():
    task = "Task040_MRAAneurysms"
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
    case_ids = [p.name for p in source_data_dir.iterdir() if p.is_dir() and p.name.startswith("sub-")]
    case_ids.sort()
    logger.info(f"Found {len(case_ids)} case ids")
    assert len(case_ids) == 284, "Missing cases"

    num_processes = int(os.getenv("det_num_threads", 4))
    if num_processes > 0:
        # multiprocess version
        logger.info(f"Using {num_processes} processes for preparation")
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(
                prepare_case,
                zip(
                    case_ids,
                    repeat(source_data_dir),
                    repeat(source_label_dir),
                    repeat(target_data_dir),
                    repeat(target_label_dir),
                ),
            )
    else:
        # main process version
        logger.info("Using main process for preparation")
        for cid in maybe_verbose_iterable(case_ids):
            logger.info(f"Preparing case {cid}")
            prepare_case(
                case_id=cid,
                source_data=source_data_dir,
                source_label_dir=source_label_dir,
                target_data_dir=target_data_dir,
                target_label_dir=target_label_dir,
            )

    create_test_split(
        splitted_dir=raw_splitted_dir,
        num_modalities=len(meta["modalities"]),
        test_size=0.3,
        random_state=0,
        shuffle=True,
        do_stratify=True,
    )


if __name__ == "__main__":
    main()
