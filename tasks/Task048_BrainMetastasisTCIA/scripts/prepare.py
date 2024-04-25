import os
import shutil
import sys
from pathlib import Path

import SimpleITK as sitk
from loguru import logger

from nndet.io.itk import load_sitk
from nndet.io.load import save_json

# from nndet.io.prepare import create_test_split
from nndet.io.prepare import create_test_split
from nndet.utils.check import env_guard
from nndet.utils.info import maybe_verbose_iterable


def prepare_case(
    case_id: str,
    source_data: Path,
    target_data_dir: Path,
    target_label_dir: Path,
) -> None:
    # copy data
    shutil.copy2(source_data / case_id / f"{case_id}-t1c.nii.gz", target_data_dir / f"{case_id}_000_0000.nii.gz")
    shutil.copy2(source_data / case_id / f"{case_id}-t1n.nii.gz", target_data_dir / f"{case_id}_000_0001.nii.gz")
    shutil.copy2(source_data / case_id / f"{case_id}-t2f.nii.gz", target_data_dir / f"{case_id}_000_0002.nii.gz")
    shutil.copy2(source_data / case_id / f"{case_id}-t2w.nii.gz", target_data_dir / f"{case_id}_000_0003.nii.gz")

    # copy seg
    seg_itk = load_sitk(source_data / case_id / f"{case_id}-seg.nii.gz")
    seg_np = sitk.GetArrayFromImage(seg_itk)
    seg_np[seg_np == 1] = 1  # map necrosis
    seg_np[seg_np == 2] = 0  # map enhancing tumor to background
    seg_np[seg_np == 3] = 1  # map tumor core
    new_seg_itk = sitk.GetImageFromArray(seg_np)
    new_seg_itk.CopyInformation(seg_itk)
    sitk.WriteImage(new_seg_itk, target_label_dir / f"{case_id}_000.nii.gz")
    # shutil.copy2(source_data / case_id / f"{case_id}-seg.nii.gz", target_label_dir / f"{case_id}_000.nii.gz")


@env_guard
def main():
    task = "Task048_BrainMetastasisTCIA"
    det_data_dir = Path(os.getenv("det_data"))
    task_data_dir = det_data_dir / task

    # setup raw paths
    source_data_dir = task_data_dir / "raw" / "Pretreat-MetsToBrain-Masks"
    if not source_data_dir.is_dir():
        raise RuntimeError(f"{source_data_dir} should contain the raw data but does not exist.")

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
        "labels": {"0": "metastasis"},
        "modalities": {"0": "T1c", "1": "T1n", "2": "T2f", "3": "T2w"},
        "dim": 3,
        # needed to run connected components for instances
        "seg2det_stuff": [],  # map the remaining classes to semantic classes
        "seg2det_things": [1],
        "min_size": 0,
        "min_vol": 0,
    }
    save_json(meta, task_data_dir / "dataset.json")

    # prepare data & label
    case_ids = [p.name for p in source_data_dir.iterdir() if p.is_dir() and p.name.startswith("BraTS-MET-")]
    case_ids.sort()
    logger.info(f"Found {len(case_ids)} case ids")
    assert len(case_ids) == 200, f"Missing cases, found {len(case_ids)} expected 200."

    for cid in maybe_verbose_iterable(case_ids):
        logger.info(f"Preparing case {cid}")
        prepare_case(
            case_id=cid,
            source_data=source_data_dir,
            target_data_dir=target_data_dir,
            target_label_dir=target_label_dir,
        )


if __name__ == "__main__":
    main()
