import os
import shutil
import sys
from pathlib import Path
from re import M
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
    mask,  # z, y, x
    idx,
    cx,
    cy,
    cz,
    dx,
    dy,
    dz,
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
    mask_itk.SetOrigin(data_itk.GetOrigin())
    mask_itk.SetDirection(data_itk.GetDirection())
    mask_itk.SetSpacing(data_itk.GetSpacing())

    # saving
    sitk.WriteImage(mask_itk, str(target_label_dir / f"{cid}.nii.gz"))
    save_json({"instances": instances}, target_label_dir / f"{cid}.json")


@env_guard
def main():
    task_name = "Task033_MelaSpace"
    det_data_dir = Path(os.getenv("det_data"))
    task_data_dir = det_data_dir / task_name

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(task_data_dir / "prepare.log", level="DEBUG")
    logger.info(f"Preparing task: {task_name}")

    # setup raw splitted dirs
    target_data_dir = task_data_dir / "raw_splitted" / "imagesTr"
    if not target_data_dir.is_dir():
        raise RuntimeError(
            "Please read the README of the prepare script, " f"required folder {target_data_dir} does not exist."
        )
    target_label_dir = task_data_dir / "raw_splitted" / "labelsTr"
    target_label_dir.mkdir(exist_ok=True, parents=True)

    label_df_path = task_data_dir / "mela_train_val_annotations.csv"
    if not label_df_path.is_file():
        raise RuntimeError(f"Expected labels file at {label_df_path}, aborting.")

    # prepare dataset info
    meta = {
        "name": "Mela",
        "task": "Task030_Mela",
        "target_class": None,
        "test_labels": False,
        "labels": {"0": "lesion"},
        "modalities": {"0": "CT"},
        "dim": 3,
    }
    save_json(meta, task_data_dir / "dataset.json")

    # prepare data & label
    case_names = [p.name for p in target_data_dir.glob("*.nii.gz")]
    case_names.sort()
    case_ids = [f"mela_{(cn).split('.')[0].split('_')[1]}" for cn in case_names]
    # case_ids = [(cn).split('.')[0] for cn in case_names]
    print(f"Found {len(case_ids)} case ids")
    print(case_ids)
    for cid, cn in maybe_verbose_iterable(zip(case_ids, case_names)):
        if len(list(cid.split("_"))) == 3:
            print(f"{cid} seems to be renamed already, skipping for now")

        os.rename(target_data_dir / f"{cn}", target_data_dir / f"{cid}_0000.nii.gz")

    label_df = pd.read_csv(label_df_path)
    print(label_df)
    for rid, cid in maybe_verbose_iterable(enumerate(case_ids)):
        logger.info(f"Processing case {cid}")

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


if __name__ == "__main__":
    main()
