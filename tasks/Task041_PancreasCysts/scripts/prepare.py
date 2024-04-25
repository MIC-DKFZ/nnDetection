import os
import shutil
import sys
from pathlib import Path

from loguru import logger

from nndet.io.load import save_json
from nndet.io.prepare import create_test_split
from nndet.utils.check import env_guard
from nndet.utils.info import maybe_verbose_iterable


def prepare_case(
    case_id: str,
    source_data: Path,
    target_data_dir: Path,
    target_label_dir: Path,
) -> None:
    # scan id _ patient id _ modality id
    shutil.copy2(source_data / f"{case_id}" / "ct.nii.gz", target_data_dir / f"pancCyst{case_id}_000_0000.nii.gz")
    shutil.copy2(
        source_data / f"{case_id}" / "cyst_mask_groundtruth.nii.gz", target_label_dir / f"pancCyst{case_id}_000.nii.gz"
    )


@env_guard
def main():
    task = "Task041_PancreasCysts"
    det_data_dir = Path(os.getenv("det_data"))
    task_data_dir = det_data_dir / task

    # setup raw paths
    source_data_dir = task_data_dir / "raw" / "Zenodo_upload" / "data"
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
        "labels": {"0": "cyst"},
        "modalities": {"0": "CT"},
        "dim": 3,
        # needed to run connected components for instances
        "seg2det_stuff": [],
        "seg2det_things": [1],
        "min_size": 0,
        "min_vol": 0,
        "min_vol_mm3": 10,  # remove all instances below 10 mm^3 according to original paper
    }
    save_json(meta, task_data_dir / "dataset.json")

    # prepare data & label
    case_ids = [p.name for p in source_data_dir.iterdir() if p.is_dir()]
    case_ids.sort()
    logger.info(f"Found {len(case_ids)} case ids")
    assert len(case_ids) == 221, "Missing cases"

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
