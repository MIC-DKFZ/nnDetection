import multiprocessing
import os
import sys
import traceback
from itertools import repeat
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from loguru import logger

from nndet.io.load import save_json
from nndet.io.prepare import create_test_split
from nndet.utils.check import env_guard
from nndet.utils.info import maybe_verbose_iterable

# only use 3 post sequences since not all patients have 4
# map nndet mod key to excel file key
SEQUENCES = {
    "pre": "pre",
    "post1": "post_1",
    "post2": "post_2",
    "post3": "post_3",
    # "t1": "T1",
}
REF = "pre"
RESAMPLE = ["T1"]


def select_folder_from_directory(directory: Path) -> Path:
    all_dirs = [p for p in directory.iterdir() if p.is_dir()]
    assert len(all_dirs) == 1, f"Expected one directory in {directory} but found {all_dirs}"
    return all_dirs[0]


def get_last_two_dirs(p: str) -> Tuple[str, str]:
    splits = p.split("/")
    last_elem = splits[-1]

    # sometimes "//" is present in the path
    second_last_elem = splits[-2]
    if second_last_elem == "":
        second_last_elem = splits[-3]

    return second_last_elem, last_elem


def get_seq_dir(case_subdir: Path, base_dir_name: Path) -> Path:
    if (case_subdir / base_dir_name).is_dir():
        return case_subdir / base_dir_name
    else:
        sel_dirs = [p for p in case_subdir.iterdir() if p.is_dir() and p.name.endswith(base_dir_name)]
        if len(sel_dirs) == 1:
            return sel_dirs[0]
        else:
            raise RuntimeError(
                f"Expected one directory in {case_subdir} with name {base_dir_name} but found {sel_dirs}"
            )


def get_dcm_paths(seq_dir: Path, rel_paths: pd.Series) -> Sequence[Path]:
    dcm_paths = []
    for rel_path in rel_paths:
        dcm_path = seq_dir / rel_path[1]
        if not dcm_path.is_file():
            rel_path_split = rel_path[1].split("-", 1)
            dcm_path = seq_dir / f"{rel_path_split[0]}-{rel_path_split[1][1:]}"
        assert dcm_path.is_file(), f"Expected file {dcm_path} does not exist"
        dcm_paths.append(dcm_path)
    return dcm_paths


def read_dcm_sitk(files: Sequence[Path]) -> sitk.Image:
    files = [str(f) for f in files]
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    image_itk = reader.Execute()
    return image_itk


def str_to_path(p: str) -> Path:
    pathlib_path = Path("")
    for _p in p.split("/"):
        pathlib_path /= _p
    return pathlib_path


def check_inverted_file_ordering(files: Sequence[Path]) -> bool:
    if len(files) < 2:
        return False

    # Load first and last DICOM files to extract position information
    first_dicom = pydicom.dcmread(str(files[0]))
    last_dicom = pydicom.dcmread(str(files[-1]))

    # Extract positions from the DICOM files
    # first_position = first_dicom.ImagePositionPatient
    # last_position = last_dicom.ImagePositionPatient
    # Check if the ordering is inverted based on z-coordinates
    # return first_position[2] > last_position[2]

    # check if ordering is inverted based on InstanceNumber
    first_position = first_dicom.InstanceNumber
    last_position = last_dicom.InstanceNumber
    # not inverted if first_position < last_position
    return first_position > last_position


def create_mask(
    df_boxes_case: pd.DataFrame, ref_itk: sitk.Image, is_inverted: bool
) -> Union[sitk.Image, Dict[int, int]]:
    ref_np = sitk.GetArrayFromImage(ref_itk)
    mask_np = np.zeros_like(ref_np, dtype=np.uint8)

    instances = {}
    for lesion_idx, (_, row) in enumerate(df_boxes_case.iterrows(), start=1):
        slicing = (
            slice(row["Start Slice"], row["End Slice"]),
            slice(row["Start Row"], row["End Row"]),
            slice(row["Start Column"], row["End Column"]),
        )
        mask_np[slicing] = lesion_idx
        instances[lesion_idx] = 0
    if is_inverted:
        mask_np = mask_np[::-1, :, :]
    mask_itk = sitk.GetImageFromArray(mask_np)
    mask_itk.CopyInformation(ref_itk)
    return mask_itk, instances


def prepare_case(
    case_id: str,
    df_case: pd.DataFrame,
    df_boxes_case: pd.DataFrame,
    source_data: Path,
    target_data_dir: Path,
    target_label_dir: Path,
) -> None:
    # process case
    for seq_idx, excel_seq in enumerate(SEQUENCES.values()):
        df_case_seq = df_case[df_case["modality_identifier"] == excel_seq].sort_values(by="original_path_and_filename")

        case_subdir = select_folder_from_directory(source_data / case_id)
        # handling of special cases in pathing ...
        seq_dir = get_seq_dir(case_subdir, df_case_seq.iloc[0]["rel_path"][0])
        dcm_paths = get_dcm_paths(seq_dir, df_case_seq["rel_path"])

        seq_itk = read_dcm_sitk(dcm_paths)
        if excel_seq == REF:
            is_inverted = check_inverted_file_ordering(dcm_paths)
            if is_inverted:
                logger.warning(f"Case {case_id} has inverted file ordering")
            mask_itk, instances = create_mask(
                df_boxes_case=df_boxes_case,
                ref_itk=seq_itk,
                is_inverted=is_inverted,
            )
            sitk.WriteImage(mask_itk, str(target_label_dir / f"{case_id}.nii.gz"))
            save_json({"instances": instances}, target_label_dir / f"{case_id}.json")
        sitk.WriteImage(seq_itk, str(target_data_dir / f"{case_id}_{seq_idx:04d}.nii.gz"))


def filter_and_prepare_case(
    case_id,
    df_image_paths,
    df_boxes,
    source_data,
    target_data_dir,
    target_label_dir,
):
    try:
        logger.info(f"Preparing case {case_id}")
        df_case = df_image_paths[df_image_paths["patient_identifier"] == case_id]
        assert len(df_case) > 0

        df_boxes_case = df_boxes[df_boxes["Patient ID"] == case_id]
        assert len(df_boxes_case) > 0  # all patients have cancer

        prepare_case(
            case_id=case_id,
            df_case=df_case,
            df_boxes_case=df_boxes_case,
            source_data=source_data,
            target_data_dir=target_data_dir,
            target_label_dir=target_label_dir,
        )
    except Exception as e:
        logger.error(f"Failed to prepare case {case_id} with {e}")
        logger.error(f"{traceback.format_exc()}")
        raise e


@env_guard
def main():
    task = "Task042_Duke"
    det_data_dir = Path(os.getenv("det_data"))
    task_data_dir = det_data_dir / task

    # setup raw paths
    raw_data_dir = task_data_dir / "raw"
    source_data_dir = task_data_dir / "raw" / "Duke-Breast-Cancer-MRI"
    if not source_data_dir.is_dir():
        raise RuntimeError(f"{source_data_dir} should contain the raw data but does not exist.")

    df_image_paths = pd.read_excel(raw_data_dir / "Breast-Cancer-MRI-filepath_filename-mapping.xlsx")
    df_image_paths["patient_identifier"] = df_image_paths["original_path_and_filename"].apply(lambda x: x.split("/")[1])
    df_image_paths["modality_identifier"] = df_image_paths["original_path_and_filename"].apply(
        lambda x: x.split("/")[2]
    )
    df_image_paths["rel_path"] = df_image_paths["descriptive_path"].apply(get_last_two_dirs)
    df_boxes = pd.read_excel(raw_data_dir / "Annotation_Boxes.xlsx")

    # setup logger
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
        "modalities": {k: i for k, i in enumerate(SEQUENCES.keys())},
        "dim": 3,
    }
    save_json(meta, task_data_dir / "dataset.json")

    # prepare data & label
    case_ids = [p.name for p in source_data_dir.iterdir() if p.is_dir()]
    case_ids.sort()
    logger.info(f"Found {len(case_ids)} case ids")
    assert len(case_ids) == 922, "Missing cases"

    # case_ids = ["Breast_MRI_001", "Breast_MRI_015"]
    num_processes = int(os.getenv("det_num_threads", 4))
    if num_processes < 1:
        # multiprocess version
        logger.info(f"Using {num_processes} processes for preparation")
        for cid in maybe_verbose_iterable(case_ids):
            filter_and_prepare_case(
                case_id=cid,
                df_image_paths=df_image_paths,
                df_boxes=df_boxes,
                source_data=source_data_dir,
                target_data_dir=target_data_dir,
                target_label_dir=target_label_dir,
            )
    else:
        logger.info(f"Using {num_processes} processes for preparation")
        with multiprocessing.Pool() as pool:
            pool.starmap(
                filter_and_prepare_case,
                zip(
                    case_ids,
                    repeat(df_image_paths),
                    repeat(df_boxes),
                    repeat(source_data_dir),
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
