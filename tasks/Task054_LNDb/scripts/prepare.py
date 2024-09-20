import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from loguru import logger

from nndet.io.itk import load_sitk
from nndet.io.load import save_json
from nndet.io.prepare import create_test_split
from nndet.utils.check import env_guard
from nndet.utils.info import maybe_verbose_iterable


def prepare_case(
    case_id: str,
    df_labels: pd.DataFrame,
    source_data: Path,
    source_label_dir: Path,
    target_data_dir: Path,
    target_label_dir: Path,
) -> None:
    logger.info(f"Preparing case {case_id}")
    label_paths = [
        p for p in source_label_dir.iterdir() if p.name.startswith(f"{case_id}_rad") and p.name.endswith(".mhd")
    ]

    data_itk = load_sitk(source_data / f"{case_id}.mhd")
    label_itk_dict = {str(p.stem[-1]): load_sitk(p) for p in label_paths}
    label_np_dict = {k: sitk.GetArrayFromImage(i) for k, i in label_itk_dict.items()}

    instances_meta = {}
    instances = {}
    mask_np = np.zeros_like(sitk.GetArrayViewFromImage(data_itk), dtype=np.uint8)

    # iterate lesions
    case_id_int = int(case_id.rsplit("-", 1)[1])
    if case_id_int in df_labels.index:
        df_case = df_labels.loc[[case_id_int]]
        df_case = df_case[df_case["Nodule"] == 1]  # filter for nodules

        for lesion_id, (_, row) in enumerate(df_case.iterrows(), start=1):
            assert lesion_id > 0, "Lesion ID must be positive"
            rad_ids = [int(i) for i in str(row["RadID"]).split(",")]
            rad_finding_ids = [int(i) for i in str(row["RadFindingID"]).split(",")]
            assert len(rad_ids) == len(rad_finding_ids), "Rad and RadFinding IDs do not match"

            # fill annotations for single lesions
            tmp_mask = np.zeros_like(mask_np)
            for rad_id, rad_finding_id in zip(rad_ids, rad_finding_ids):
                tmp_mask += label_np_dict[str(rad_id)] == rad_finding_id
            tmp_mask = tmp_mask >= 1

            assert mask_np[tmp_mask].sum() == 0, "Overlapping masks"
            mask_np[tmp_mask] = lesion_id
            instances[lesion_id] = 0
            instances_meta[lesion_id] = {
                "agr_level": row["AgrLevel"],
                "text": row["Text"],
                "volume": row["Volume"],
                "FindingID": row["FindingID"],
                "Nodule": row["Nodule"],
            }
    else:
        logger.info(f"No annotations for case {case_id} in csv")
        for _, item in label_np_dict.items():
            assert item.max() == 0
    assert mask_np.max() == len(instances), "Missing instances"

    mask_itk = sitk.GetImageFromArray(mask_np)
    mask_itk.CopyInformation(data_itk)

    # data
    sitk.WriteImage(data_itk, target_data_dir / f"{case_id}_0000.nii.gz")

    # labels
    sitk.WriteImage(mask_itk, target_label_dir / f"{case_id}.nii.gz")
    save_json({"instances": instances, "meta": instances_meta}, target_label_dir / f"{case_id}.json")


@env_guard
def main():
    task = "Task054_LNDb"
    det_data_dir = Path(os.getenv("det_data"))
    task_data_dir = det_data_dir / task

    # setup raw paths
    source_data_dir = task_data_dir / "raw" / "data"
    source_label_dir = task_data_dir / "raw" / "masks"
    df_path = task_data_dir / "raw" / "trainNodules_gt.csv"
    if not source_data_dir.is_dir():
        raise RuntimeError(f"{source_data_dir} should contain the raw data but does not exist.")
    if not source_label_dir.is_dir():
        raise RuntimeError(f"{source_label_dir} should contain the raw labels but does not exist.")
    if not df_path.is_file():
        raise RuntimeError(f"{df_path} should contain the raw labels csv but does not exist.")

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
        "labels": {"0": "nodule"},
        "modalities": {"0": "CT"},
        "annotation_style": "seg",
    }
    save_json(meta, task_data_dir / "dataset.json")

    # prepare data & label
    case_ids = list(set([p.stem for p in source_data_dir.iterdir() if p.name.endswith(".mhd")]))
    case_ids.sort()
    logger.info(f"Found {len(case_ids)} case ids")
    assert len(case_ids) == 236, "Missing cases"

    # df with LNDbID	RadID	RadFindingID	FindingID	x	y	z	AgrLevel	Nodule	Volume	Text
    df_labels = pd.read_csv(df_path, index_col="LNDbID")

    for cid in maybe_verbose_iterable(case_ids):
        logger.info(f"Preparing case {cid}")
        prepare_case(
            case_id=cid,
            df_labels=df_labels,
            source_data=source_data_dir,
            source_label_dir=source_label_dir,
            target_data_dir=target_data_dir,
            target_label_dir=target_label_dir,
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
