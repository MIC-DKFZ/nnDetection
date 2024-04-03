import os
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from nndet.io import save_json
from nndet.io.itk import load_sitk
from nndet.utils.check import env_guard


@env_guard
def main():
    # task = "Task053_PN9"
    # det_data_dir = Path(os.getenv("det_data"))

    task = "Task029_PN9"
    det_data_dir = Path("/media/E130-Personal/Baumgartner/nndet_data")

    task_data_dir = det_data_dir / task

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(task_data_dir / "csv_to_mitk.log", level="DEBUG")

    # csv path
    images_dir = task_data_dir / "raw_splitted" / "imagesTr"
    images_dir.mkdir(exist_ok=True, parents=True)
    csv_path = task_data_dir / "raw" / "train" / "train_anno.csv"
    save_dir = task_data_dir / "raw" / "train_boxes_mitk"
    save_dir.mkdir(exist_ok=True, parents=True)

    # dataframe with cols: pid, nodule_class, xmin, xmax, nodule_id
    # index start at one (!)
    df = pd.read_csv(csv_path)

    # zero index
    for ax in ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]:
        df[ax] = df[ax] - 1

    for idx, (pid, group) in enumerate(df.groupby("pid")):
        logger.info(f"Processing {pid}")
        pid_str_format = f"{pid:05d}"
        data_itk = load_sitk(images_dir / f"{pid_str_format}_0000.nii.gz")
        size = list(data_itk.GetSize())

        json_data = {
            "FileFormat": "MITK ROI",
            "Version": 1,
            "Caption": "{ID}: {class}",
            "Geometry": {"Origin": [0.0, 0.0, 0.0], "Spacing": [1.0, 1.0, 1.0], "Size": size},
            "ROIs": [],
        }

        for _, row in group.iterrows():
            roi = {
                "ID": int(row["nodule_id"]),
                "Min": [row["xmin"], row["ymin"], row["zmin"]],
                "Max": [row["xmax"], row["ymax"], row["zmax"]],
                "Properties": {"ColorProperty": {"color": [0, 0, 1]}, "StringProperty": {"class": row["nodule_class"]}},
            }
            json_data["ROIs"].append(roi)

        save_json(json_data, save_dir / f"{pid_str_format}.json")

        if idx > 700:
            break


if __name__ == "__main__":
    main()
