import os
import shutil
import sys
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from nndet.utils.check import env_guard

import numpy as np
from loguru import logger
import SimpleITK as sitk

from nndet.io import save_json
from nndet.io.prepare import create_test_split
from nndet.io.itk import load_sitk_as_array
from nndet.utils.info import maybe_verbose_iterable


def prepare_image(
        case_id: str,
        base_dir: Path,
        mask_dir: Path,
        raw_splitted_dir: Path,
):
    logger.info(f"Processing {case_id}")
    root_data_dir = base_dir / case_id
    patient_data_dir = []
    for root, dirs, files in os.walk(root_data_dir, topdown=False):
        if any([f.endswith(".dcm") for f in files]):
            patient_data_dir.append(Path(root))
    assert len(patient_data_dir) == 1
    patient_data_dir = patient_data_dir[0]

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(patient_data_dir))
    reader.SetFileNames(dicom_names)
    data_itk = reader.Execute()

    patient_label_dir = mask_dir / case_id
    label_path = [p for p in patient_label_dir.iterdir() if p.is_file() and p.name.endswith(".nii.gz")]
    assert len(label_path) == 1
    label_path = label_path[0]

    mask = load_sitk_as_array(label_path)[0]
    instances = np.unique(mask)
    instances = instances[instances > 0]
    meta = {"instances": {str(int(i)): 0 for i in instances}}
    meta["original_path_data"] = str(patient_data_dir)
    meta["original_path_label"] = str(label_path)

    save_json(meta, raw_splitted_dir / "labelsTr" / f"{case_id}.json")

    sitk.WriteImage(data_itk, str(raw_splitted_dir / "imagesTr" / f"{case_id}_0000.nii.gz"))
    shutil.copy(label_path, raw_splitted_dir / "labelsTr" / f"{case_id}.nii.gz")



@env_guard
def main():
    det_data_dir = Path(os.getenv("det_data"))
    task_data_dir = det_data_dir / "Task025_LymphNodes"
    source_data_base = task_data_dir / "raw"
    if not source_data_base.is_dir():
        raise RuntimeError(f"{source_data_base} should contain the raw data but does not exist.")

    raw_splitted_dir = task_data_dir / "raw_splitted"
    (raw_splitted_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (raw_splitted_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    (raw_splitted_dir / "imagesTs").mkdir(parents=True, exist_ok=True)
    (raw_splitted_dir / "labelsTs").mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, format="{level} {message}", level="DEBUG")
    logger.add(raw_splitted_dir.parent / "prepare.log", level="DEBUG")

    meta = {
        "name": "Lymph Node TCIA",
        "task": "Task025_LymphNodes",

        "target_class": None,
        "test_labels": True,

        "labels": {
            "0": "LymphNode",
        },
        "modalities": {
            "0": "CT",
        },
        "dim": 3,
    }

    save_json(meta, raw_splitted_dir.parent / "dataset.json")

    base_dir = source_data_base / "CT Lymph Nodes"
    mask_dir = source_data_base / "MED_ABD_LYMPH_MASKS"

    case_ids = sorted([p.name for p in base_dir.iterdir() if p.is_dir()])
    logger.info(f"Found {len(case_ids)} cases in {base_dir}")

    for cid in maybe_verbose_iterable(case_ids):
        prepare_image(
            case_id=cid,
            base_dir=base_dir,
            mask_dir=mask_dir,
            raw_splitted_dir=raw_splitted_dir,
        )

    # with Pool(processes=6) as p:
    #     p.starmap(
    #         prepare_image,
    #         zip(
    #             case_ids,
    #             repeat(base_dir),
    #             repeat(mask_dir),
    #             repeat(raw_splitted_dir)
    #         )
    #     )

    create_test_split(raw_splitted_dir,
                      num_modalities=len(meta["modalities"]),
                      test_size=0.3,
                      random_state=0,
                      shuffle=True,
                      )


if __name__ == '__main__':
    main()
