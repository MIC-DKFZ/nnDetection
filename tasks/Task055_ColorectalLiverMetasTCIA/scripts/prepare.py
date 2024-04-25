import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pydicom
import pydicom_seg
import SimpleITK as sitk
from loguru import logger

from nndet.io.load import save_json
from nndet.io.prepare import create_test_split
from nndet.utils.check import env_guard
from nndet.utils.info import maybe_verbose_iterable

# exlude cases
exclude_cases = [
    # overlapping
    "CRLM-CT-1020",
    "CRLM-CT-1026",
    "CRLM-CT-1027",
    "CRLM-CT-1031",
    "CRLM-CT-1037",
    "CRLM-CT-1049",
    "CRLM-CT-1053",
    "CRLM-CT-1057",
    "CRLM-CT-1070",
    "CRLM-CT-1078",
    "CRLM-CT-1080",
    "CRLM-CT-1081",
    "CRLM-CT-1083",
    "CRLM-CT-1088",
    "CRLM-CT-1112",
    "CRLM-CT-1122",
    "CRLM-CT-1127",
    "CRLM-CT-1133",
    "CRLM-CT-1139",
    "CRLM-CT-1145",
    "CRLM-CT-1155",
    "CRLM-CT-1168",
    "CRLM-CT-1173",
    "CRLM-CT-1186",
    "CRLM-CT-1190",
    # missing data (at least with my download :) )
    "CRLM-CT-1183",
]


def select_folder_from_directory(directory: Path) -> Path:
    all_dirs = [p for p in directory.iterdir() if p.is_dir()]
    assert len(all_dirs) == 1, f"Expected one directory in {directory} but found {all_dirs}"
    return all_dirs[0]


def assert_coord_sys(img0_itk: sitk.Image, img1_itk: sitk.Image):
    assert img0_itk.GetSpacing() == img1_itk.GetSpacing()
    assert img0_itk.GetOrigin() == img1_itk.GetOrigin()
    assert img0_itk.GetDirection() == img1_itk.GetDirection()


def prepare_case(
    case_id: str,
    source_data: Path,
    target_data_dir: Path,
    target_label_dir: Path,
) -> None:
    source_case_dir = select_folder_from_directory(source_data / case_id)
    subdirs = [p for p in source_case_dir.iterdir() if p.is_dir()]
    assert len(subdirs) == 2, f"Expected two subdirs in {source_case_dir} but found {subdirs}"

    if "Segmentation" in subdirs[0].name:
        seg_dir = subdirs[0]
        img_dir = subdirs[1]
    elif "Segmentation" in subdirs[1].name:
        seg_dir = subdirs[1]
        img_dir = subdirs[0]
    else:
        raise RuntimeError(f"Could not find segmentation dir in {source_case_dir}")

    # load data and convert
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(img_dir))
    reader.SetFileNames(dicom_names)
    img_itk = reader.Execute()
    img_np = sitk.GetArrayFromImage(img_itk)

    dcm = pydicom.dcmread(str(seg_dir / "1-1.dcm"))
    reader = pydicom_seg.SegmentReader()
    result = reader.read(dcm)

    # create mask
    mask_np = np.zeros_like(img_np, dtype=np.uint8)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(sitk.Transform())
    resampler.SetReferenceImage(img_itk)

    lesion_idx = 1
    instances = {}
    for segment_key, segment in result.segment_infos.items():
        if "tumor" in segment.SegmentLabel.lower():
            segment_itk = result.segment_image(segment_key)
            resampled_segment_itk = resampler.Execute(segment_itk)
            assert_coord_sys(img_itk, resampled_segment_itk)
            segment_data = sitk.GetArrayFromImage(resampled_segment_itk)
            segment_bin_mask = segment_data == 1

            assert segment_bin_mask.max() == 1  # binary mask with one object
            assert mask_np[segment_bin_mask].max() == 0  # no overlap with other lesions
            mask_np[segment_data == 1] = lesion_idx
            instances[lesion_idx] = 0
            lesion_idx += 1
    mask_itk = sitk.GetImageFromArray(mask_np)
    mask_itk.CopyInformation(img_itk)

    sitk.WriteImage(img_itk, target_data_dir / f"{case_id}_0000.nii.gz")
    sitk.WriteImage(mask_itk, target_label_dir / f"{case_id}.nii.gz")
    save_json({"instances": instances}, target_label_dir / f"{case_id}.json")


@env_guard
def main():
    task = "Task055_ColorectalLiverMetasTCIA"
    det_data_dir = Path(os.getenv("det_data"))
    task_data_dir = det_data_dir / task

    # setup raw paths
    source_data_dir = task_data_dir / "raw" / "Colorectal-Liver-Metastases"
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
        "dim": 3,
        "target_class": None,
        "test_labels": True,
        "labels": {"0": "tumor"},
        "modalities": {"0": "CT"},
    }
    save_json(meta, task_data_dir / "dataset.json")

    # prepare data & label
    case_ids = [p.name for p in source_data_dir.iterdir() if p.is_dir() and p.name.startswith("CRLM-CT-")]
    case_ids_filtered = list(set(case_ids) - set(exclude_cases))
    case_ids_filtered.sort()
    logger.info(f"Found {len(case_ids)} case ids before filtering and {len(case_ids_filtered)} after filtering.")
    assert len(case_ids) == 197, "Missing cases"
    assert len(case_ids_filtered) == (197 - len(exclude_cases)), "Missing cases"

    for cid in maybe_verbose_iterable(case_ids_filtered):
        logger.info(f"Preparing case {cid}")
        prepare_case(
            case_id=cid,
            source_data=source_data_dir,
            target_data_dir=target_data_dir,
            target_label_dir=target_label_dir,
        )

    create_test_split(
        splitted_dir=raw_splitted_dir,
        num_modalities=len(meta["modalities"]),
        test_size=0.3,
        random_state=0,
        shuffle=True,
        do_stratify=True,
    )


if __name__ == "__main__":
    main()
