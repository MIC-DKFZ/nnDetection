import os
import shutil
import sys
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from nndet.io.load import save_json
from nndet.utils.check import env_guard
from nndet.utils.info import maybe_verbose_iterable

# we use these to check that the label is consistent
LABEL_MAPPER = {
    "PDAC": True,
    "non-PDAC": False,
}
LEVEL_MAPPER = {
    "radiology": False,
    "pathology": False,
    "cytology": False,
    "MSD_dataset": True,
    "NIH_dataset": False,
    "histopathology": False,
    "radiology / 3yFU": False,
}
EXCLUDE = [
    "100051_00001",
    "100036_00001",
    "100433_00001",
    "101381_00001",
]


def prepare_case(
    case_id: str,
    source_data: Path,
    source_label_dir: Path,
    target_data_dir: Path,
    target_label_dir: Path,
    patient_meta: pd.Series,
) -> None:
    # scan id _ patient id _ modality id
    shutil.copy2(source_data / f"{case_id}_0000.nii.gz", target_data_dir / f"{case_id}_0000.nii.gz")
    shutil.copy2(source_label_dir / f"{case_id}.nii.gz", target_label_dir / f"{case_id}.nii.gz")

    if patient_meta["label"] == "non-PDAC":
        label_int = 0
    elif patient_meta["label"] == "PDAC":
        label_int = 1
    else:
        raise RuntimeError(f"Unknown label {patient_meta['patient_label']}")

    meta = {
        "instances": "UNKNOWN",
        "clinical": {
            "date": str(patient_meta["anonymized_study_date"]),
            "age": str(patient_meta["patient_age"]),
            "patient_sex": str(patient_meta["patient_sex"]),
            "scanner": str(patient_meta["scanner"]),
            "patient_label": str(patient_meta["label"]),
            "patient_label_int": label_int,
            "level": str(patient_meta["level"]),
        },
    }
    save_json(meta, target_label_dir / f"{case_id}.json")


def create_custom_split(
    patient_info: pd.DataFrame,
    splitted_dir: os.PathLike,
    num_modalities: int,
    test_size: float = 0.3,
) -> None:
    # dataframe with "label_bool", "PANORAMA_patient_id"
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

    case_ids = list(patient_info.index)
    case_ids.sort()

    patient_info_grouped = patient_info.groupby("PANORAMA_patient_id").max()
    patient_ids = sorted(patient_info_grouped.index)
    patient_ids = list(set(patient_ids))
    patient_ids.sort()
    stratify = [patient_info_grouped.loc[pid]["label_bool"] for pid in patient_ids]

    logger.info(f"Found {len(case_ids)} cases and {len(patient_ids)} patients to split")
    train_pids, test_pids = train_test_split(
        patient_ids,
        test_size=test_size,
        random_state=0,
        shuffle=True,
        stratify=stratify,
    )
    logger.info(f"Using {train_pids} patients for training and {test_pids} patients for testing.")
    logger.info("Moving data ...")
    for cid in case_ids:
        if patient_info.loc[cid]["PANORAMA_patient_id"] in test_pids:
            for modality in range(num_modalities):
                shutil.move(
                    images_tr / f"{cid}_{modality:04d}.nii.gz",
                    images_ts / f"{cid}_{modality:04d}.nii.gz",
                )
            for p in labels_tr.glob(f"{cid}*"):
                shutil.move(p, labels_ts / p.name)
    return train_pids, test_pids


@env_guard
def main():
    task = "Task056_PanoramaSubset"
    det_data_dir = Path(os.getenv("det_data"))
    task_data_dir = det_data_dir / task

    # setup raw paths
    source_data_dir = task_data_dir / "raw" / "imagesTr"
    source_label_dir = task_data_dir / "raw" / "labelsTr"
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
        "target_class": None,
        "test_labels": True,
        "labels": {"0": "tumor"},
        "modalities": {"0": "CT"},
        "dim": 3,
        # Multiple scans per patient available
        "session_id": True,
        # needed to run connected components for instances
        "seg2det_stuff": [2, 3, 4, 5, 6],
        "seg2det_things": [1],
        "min_size": 0,
        "min_vol": 0,
    }
    save_json(meta, task_data_dir / "dataset.json")

    # load patient meta information
    pano_labels_dir = task_data_dir / "raw" / "panorama_labels"
    assert pano_labels_dir.is_dir(), f"{pano_labels_dir} does not exist"
    patient_df = pd.read_excel(pano_labels_dir / "clinical_information.xlsx")
    patient_df = patient_df.set_index("PANORAMA_study_id")
    patient_df["label_bool"] = patient_df["label"].map(LABEL_MAPPER)
    patient_df["level_msd"] = patient_df["level"].map(LEVEL_MAPPER)

    # prepare data & label
    image_cases_found = [
        p.name.rsplit(".", 2)[0].rsplit("_", 1)[0] for p in source_data_dir.iterdir() if p.name.endswith(".nii.gz")
    ]
    image_cases_found.sort()
    assert len(image_cases_found) == 2235, "Inconsistent cases"

    # filter patients
    cases_automatic_labels = [
        p.name.rsplit(".", 2)[0] for p in (pano_labels_dir / "automatic_labels").iterdir() if p.name.endswith(".nii.gz")
    ]
    cases_automatic_labels.sort()

    filtered_series = []
    for row in patient_df.iterrows():
        if row[0] in EXCLUDE:
            continue
        if row[0] in cases_automatic_labels and row[1]["label_bool"]:  # exclude all automatic labels of PDAC cases
            continue
        if row[1]["level_msd"]:  # exclude all MSD cases
            continue
        filtered_series.append(row[1])
    patient_filtered_df = pd.DataFrame(filtered_series)

    for cid in maybe_verbose_iterable(patient_filtered_df.index):
        logger.info(f"Preparing case {cid}")
        patient_meta = patient_df.loc[cid]
        prepare_case(
            case_id=cid,
            source_data=source_data_dir,
            source_label_dir=source_label_dir,
            target_data_dir=target_data_dir,
            target_label_dir=target_label_dir,
            patient_meta=patient_meta,
        )

    # create custom split
    create_custom_split(
        patient_info=patient_filtered_df,
        splitted_dir=raw_splitted_dir,
        num_modalities=len(meta["modalities"]),
        test_size=0.3,
    )


if __name__ == "__main__":
    main()
