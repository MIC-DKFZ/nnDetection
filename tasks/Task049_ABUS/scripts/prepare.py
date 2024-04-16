import multiprocessing
import os
import shutil
import sys
from itertools import repeat
from pathlib import Path

import SimpleITK as sitk
from loguru import logger

from nndet.io.itk import load_sitk
from nndet.io.load import save_json
from nndet.io.prepare import create_test_split
from nndet.utils.check import env_guard
from nndet.utils.info import maybe_verbose_iterable


def prepare_case(
    case_id: str,
    raw_images: Path,
    raw_labels: Path,
    target_images: Path,
    target_labels: Path,
) -> None:
    logger.info(f"Prepare case {case_id}")
    image_itk = load_sitk(raw_images / f"DATA_{case_id}.nrrd")
    label_itk = load_sitk(raw_labels / f"MASK_{case_id}.nrrd")

    label_np = sitk.GetArrayFromImage(label_itk)
    if label_np.max() == 0:
        info_json = {"instances": {}}
    elif label_np.max() == 1:
        info_json = {"instances": {1: 0}}
    else:
        logger.error(f"ERROR case: {case_id}")
        raise RuntimeError("Unknown label values")

    sitk.WriteImage(image_itk, str(target_images / f"abus_{case_id}_0000.nii.gz"))
    sitk.WriteImage(label_itk, str(target_labels / f"abus_{case_id}.nii.gz"))
    save_json(info_json, target_labels / f"abus_{case_id}.json")


@env_guard
def main():
    task = "Task049_ABUS"
    det_data_dir = Path(os.getenv("det_data"))
    task_data_dir = det_data_dir / task

    # setup raw paths
    source_data_dir = task_data_dir / "raw" / "data"
    source_label_dir = task_data_dir / "raw" / "MASK"
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
        "test_labels": False,
        "labels": {"0": "tumor"},
        "modalities": {"0": "ABUS"},
    }
    save_json(meta, task_data_dir / "dataset.json")

    # prepare data & label
    case_ids = [p.stem.rsplit("_", 1)[1] for p in source_data_dir.glob("*.nrrd")]
    case_ids.sort()
    print(f"Found {case_ids}")

    assert len(case_ids) == 100, "Missing cases"

    num_processes = int(os.getenv("det_num_threads", 4))
    if num_processes < 1:
        # multiprocess version
        logger.info(f"Only using main process for preparation")
        for cid in maybe_verbose_iterable(case_ids):
            prepare_case(
                case_id=cid,
                raw_images=source_data_dir,
                raw_labels=source_label_dir,
                target_images=target_data_dir,
                target_labels=target_label_dir,
            )
    else:
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

    # create_test_split(
    #     splitted_dir=raw_splitted_dir,
    #     num_modalities=len(meta["modalities"]),
    #     test_size=0.3,
    #     random_state=0,
    #     shuffle=True,
    # )


if __name__ == "__main__":
    main()
