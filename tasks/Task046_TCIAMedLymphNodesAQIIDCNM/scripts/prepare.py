import os
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from loguru import logger

from nndet.io import save_json
from nndet.io.itk import load_sitk_as_array
from nndet.io.prepare import create_test_split
from nndet.utils.check import env_guard

MIN_DCM_FILES = 5


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

        if len([f.endswith(".dcm") for f in files]) >= MIN_DCM_FILES:
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

    raw_data_dir = det_data_dir / "Task046_TCIAMedLymphNodesAQIIDCNM" / "raw"
    raw_data_subdirs = [p for p in raw_data_dir.iterdir() if (p.is_dir() and p.name.startswith("TCIA_CT_Lymph_Nodes"))]
    assert len(raw_data_subdirs) == 1
    source_data_tcia = raw_data_subdirs[0] / "CT Lymph Nodes"
    source_masks_tcia = raw_data_dir / "MED_ABD_LYMPH_MASKS"

    for t, prefix in [
        ("Task046_TCIAMedLymphNodesAQIIDCNM", "MED"),
        ("Task047_TCIAAbdLymphNodesAQIIDCNM", "ABD"),
    ]:
        task_data_dir = det_data_dir / t

        logger.remove()
        logger.add(sys.stdout, level="INFO")
        logger.add(task_data_dir / "prepare.log", level="DEBUG")

        logger.info(f"Using task dir {task_data_dir}")

        # setup raw splitted dirs
        raw_splitted_dir = task_data_dir / "raw_splitted"
        target_data_dir = raw_splitted_dir / "imagesTr"
        target_data_dir.mkdir(exist_ok=True, parents=True)
        target_label_dir = raw_splitted_dir / "labelsTr"
        target_label_dir.mkdir(exist_ok=True, parents=True)

        # prepare dataset info
        meta = {
            "task": t,
            "target_class": 0,
            "test_labels": False,
            "labels": {"0": "LymphNode"},
            "modalities": {"0": "CT"},
            "dim": 3,
        }
        save_json(meta, task_data_dir / "dataset.json")

        case_ids = sorted([p.name for p in source_data_tcia.iterdir() if p.is_dir() if p.name.startswith(prefix)])
        logger.info(f"Found {len(case_ids)} cases in {source_data_tcia}")
        logger.info(f"Preparing case ids: {case_ids}")

        for cid in case_ids:
            prepare_image(
                case_id=cid,
                base_dir=source_data_tcia,
                mask_dir=source_masks_tcia,
                raw_splitted_dir=raw_splitted_dir,
            )

        create_test_split(
            raw_splitted_dir,
            num_modalities=len(meta["modalities"]),
            test_size=0.3,
            random_state=0,
            shuffle=True,
        )


if __name__ == "__main__":
    main()
