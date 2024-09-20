import multiprocessing
import os
import shutil
import sys
from itertools import repeat
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import SimpleITK as sitk
from loguru import logger

from nndet.io import save_json
from nndet.io.itk import load_sitk
from nndet.utils.check import env_guard
from nndet.utils.info import maybe_verbose_iterable


def add_ellipsoid(
    mask: np.ndarray,  # z, y, x
    idx: int,
    cx: int,
    cy: int,
    cz: int,
    dx: int,
    dy: int,
    dz: int,
) -> np.ndarray:
    assert idx > 0

    dx2 = dx / 2.0
    assert dx2 >= 0

    dy2 = dy / 2.0
    assert dy2 >= 0

    dz2 = dz / 2.0
    assert dz2 >= 0

    added = False
    for x in range(int(cx - dx2) - 2, int(cx + dx2) + 2):
        for y in range(int(cy - dy2) - 2, int(cy + dy2) + 2):
            for z in range(int(cz - dz2) - 2, int(cz + dz2) + 2):

                if (((x - cx) ** 2 / (dx2**2)) + ((y - cy) ** 2 / (dy2**2)) + ((z - cz) ** 2 / (dz2**2))) <= 1.0:
                    if x >= mask.shape[2] or y >= mask.shape[1] or z >= mask.shape[0]:
                        logger.warning("Box outside of Scan!")
                        continue

                    if mask[z, y, x] > 0 and not added:
                        logger.warning("Detected overlap!")

                    mask[z, y, x] = idx
                    added = True
    if not added:
        logger.error("INSTANCE NOT ADDED")
    return mask


def run_prep(
    cid: str,
    source_data: Path,
    target_label_dir: Path,
    boxes_mela_format: List[Dict[str, int]],
) -> None:
    data_itk = load_sitk(source_data / f"{cid}_0000.nii.gz")
    data_np = sitk.GetArrayFromImage(data_itk)

    mask_np = np.zeros_like(data_np)

    instances = {}
    for idx, entry in enumerate(boxes_mela_format, start=1):
        mask_np = add_ellipsoid(
            mask=mask_np,
            idx=idx,
            cx=entry["cx"],
            cy=entry["cy"],
            cz=entry["cz"],
            dx=entry["dx"],
            dy=entry["dy"],
            dz=entry["dz"],
        )
        instances[int(idx)] = 0

    mask_itk = sitk.GetImageFromArray(mask_np)
    mask_itk.CopyInformation(data_itk)

    # saving
    sitk.WriteImage(mask_itk, str(target_label_dir / f"{cid}.nii.gz"))
    save_json({"instances": instances}, target_label_dir / f"{cid}.json")


def filter_and_prep_case(
    cid: str,
    label_df: pd.DataFrame,
    source_data: Path,
    target_data_dir: Path,
    target_label_dir: Path,
) -> None:
    logger.info(f"Processing case {cid}")
    # data
    shutil.copy2(source_data / f"{cid}.nii.gz", target_data_dir / f"{cid}_0000.nii.gz")

    # label
    case_df = label_df[label_df["public_id"] == cid]
    temp = case_df.to_dict()
    boxes_mela_format = []

    for i in temp["public_id"].keys():
        boxes_mela_format.append(
            {
                "cx": temp["coordX"][i],
                "cy": temp["coordY"][i],
                "cz": temp["coordZ"][i],
                "dx": temp["x_length"][i],
                "dy": temp["y_length"][i],
                "dz": temp["z_length"][i],
            }
        )
    run_prep(
        cid=cid,
        source_data=target_data_dir,
        target_label_dir=target_label_dir,
        boxes_mela_format=boxes_mela_format,
    )


@env_guard
def main():
    task_name = "Task050_MELA"
    det_data_dir = Path(os.getenv("det_data"))

    task_data_dir = det_data_dir / task_name
    raw_data_dir = task_data_dir / "raw"

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(task_data_dir / "prepare.log", level="DEBUG")
    logger.info(f"Preparing task: {task_name}")

    # setup raw splitted dirs
    source_data_tr = raw_data_dir / "imagesTr"
    source_data_ts = raw_data_dir / "imagesTs"
    if not source_data_tr.is_dir():
        raise RuntimeError(f"{source_data_tr} should contain the raw data but does not exist.")
    if not source_data_ts.is_dir():
        raise RuntimeError(f"{source_data_ts} should contain the raw data but does not exist.")

    label_df_path = raw_data_dir / "mela_train_val_annotations.csv"
    if not label_df_path.is_file():
        raise RuntimeError(f"Expected labels file at {label_df_path}, aborting.")
    label_df = pd.read_csv(label_df_path)

    target_data_tr = task_data_dir / "raw_splitted" / "imagesTr"
    target_data_tr.mkdir(exist_ok=True, parents=True)
    target_label_tr = task_data_dir / "raw_splitted" / "labelsTr"
    target_label_tr.mkdir(exist_ok=True, parents=True)

    target_data_ts = task_data_dir / "raw_splitted" / "imagesTs"
    target_data_ts.mkdir(exist_ok=True, parents=True)
    target_label_ts = task_data_dir / "raw_splitted" / "labelsTs"
    target_label_ts.mkdir(exist_ok=True, parents=True)

    # prepare dataset info
    meta = {
        "task": task_name,
        "dim": 3,
        "target_class": None,
        "test_labels": True,
        "labels": {"0": "lesion"},
        "modalities": {"0": "CT"},
        "annotation_style": "weak",
    }
    save_json(meta, task_data_dir / "dataset.json")

    for source_data, target_data, target_label in zip(
        [source_data_tr, source_data_ts],
        [target_data_tr, target_data_ts],
        [target_label_tr, target_label_ts],
    ):
        logger.info("------------------------------------")
        logger.info(f"Processing data in {source_data}...")
        logger.info("------------------------------------")

        # prepare data & label
        case_ids = [p.name.rsplit(".", 2)[0] for p in source_data.glob("*.nii.gz")]
        case_ids.sort()
        logger.info(f"Found {len(case_ids)} case ids")
        logger.info(case_ids)

        num_processes = int(os.getenv("det_num_threads", 4))
        if num_processes < 1:
            # multiprocess version
            logger.info(f"Only using main process for preparation")
            for cid in maybe_verbose_iterable(case_ids):
                filter_and_prep_case(
                    cid=cid,
                    label_df=label_df,
                    source_data=source_data,
                    target_data_dir=target_data,
                    target_label_dir=target_label,
                )
        else:
            logger.info(f"Using {num_processes} processes for preparation")
            with multiprocessing.Pool(processes=num_processes) as pool:
                pool.starmap(
                    filter_and_prep_case,
                    zip(
                        case_ids,
                        repeat(label_df),
                        repeat(source_data),
                        repeat(target_data),
                        repeat(target_label),
                    ),
                )


if __name__ == "__main__":
    main()
