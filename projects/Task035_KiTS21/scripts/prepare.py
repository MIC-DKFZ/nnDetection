import os
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from loguru import logger

from nndet.io import load_sitk, save_json
from nndet.utils.check import env_guard
from nndet.utils.info import maybe_verbose_iterable


def check_itk(data_itk: sitk.Image, seg_itk: sitk.Image):
    assert np.allclose(data_itk.GetSpacing(), seg_itk.GetSpacing())
    assert np.allclose(data_itk.GetOrigin(), seg_itk.GetOrigin())
    assert np.allclose(data_itk.GetDirection(), seg_itk.GetDirection())


def run_prep(
    case_id: str,
    source_data: Path,
    target_data_dir: Path,
    target_label_dir: Path,
):
    logger.info(f"Processing case {case_id}")
    case_dir = source_data / case_id
    case_segmentation_dir = case_dir / "segmentations"

    # load data
    data_path = case_dir / "imaging.nii.gz"
    data_itk = load_sitk(data_path)
    data_np = sitk.GetArrayFromImage(data_itk)
    assert data_np.ndim == 3

    mask_np = np.zeros(data_np.shape).astype(np.uint16)
    instances = {}
    instance_id = 1

    # write instances into mask
    tumor_files = [p for p in case_segmentation_dir.glob("tumor_*")]
    num_tumor = len(tumor_files) // 3
    assert len(tumor_files) % 3 == 0

    for tumor_idx in range(num_tumor):
        tumor_stack = []

        for annotator_idx in range(3):
            tumor_file = (
                f"tumor_instance-{tumor_idx + 1}_annotation-{annotator_idx + 1}.nii.gz"
            )
            tumor_path = case_segmentation_dir / tumor_file
            tumor_itk = load_sitk(tumor_path)
            check_itk(data_itk, tumor_itk)
            tumor_stack.append(sitk.GetArrayFromImage(tumor_itk))
        tumor_stack = np.stack(tumor_stack, axis=0)
        assert tumor_stack.max() < 2, "Seg is not a binary file"
        tumor_stack = tumor_stack.sum(axis=0)
        tumor_mask = tumor_stack >= 2  # majority voting

        if mask_np[tumor_mask].sum() > 0:
            logger.warning(f"Overlapping instance {instance_id} in case {case_id}")
        mask_np[tumor_mask] = instance_id
        instances[instance_id] = 0

        instance_id = instance_id + 1

    cyst_files = [p for p in case_segmentation_dir.glob("cyst_*")]
    num_cyst = len(cyst_files) // 3
    assert len(cyst_files) % 3 == 0

    for cyst_idx in range(num_cyst):
        cyst_stack = []

        for annotator_idx in range(3):
            cyst_file = (
                f"cyst_instance-{cyst_idx + 1}_annotation-{annotator_idx + 1}.nii.gz"
            )
            cyst_path = case_segmentation_dir / cyst_file
            cyst_itk = load_sitk(cyst_path)
            check_itk(data_itk, cyst_itk)
            cyst_stack.append(sitk.GetArrayFromImage(cyst_itk))
        cyst_stack = np.stack(cyst_stack, axis=0)
        assert cyst_stack.max() < 2, "Seg is not a binary file"
        cyst_stack = cyst_stack.sum(axis=0)
        cyst_mask = cyst_stack >= 2  # majority voting

        if mask_np[cyst_mask].sum() > 0:
            logger.warning(f"Overlapping instance {instance_id} in case {case_id}")
        mask_np[cyst_mask] = instance_id
        instances[instance_id] = 1

        instance_id = instance_id + 1

    mask_itk = sitk.GetImageFromArray(mask_np)
    mask_itk.SetOrigin(data_itk.GetOrigin())
    mask_itk.SetSpacing(data_itk.GetSpacing())
    mask_itk.SetDirection(data_itk.GetDirection())

    # save files
    assert len(instances) == (num_cyst + num_tumor)
    logger.info(
        f"Generated mask {case_id} with {num_tumor} tumors and {num_cyst} cysts"
    )

    save_id = "c" + case_id.rsplit("_", 1)[1]

    sitk.WriteImage(data_itk, str(target_data_dir / f"{save_id}_0000.nii.gz"))
    sitk.WriteImage(mask_itk, str(target_label_dir / f"{save_id}.nii.gz"))
    save_json({"instances": instances}, target_label_dir / f"{save_id}")


@env_guard
def main():
    det_data_dir = Path(os.getenv("det_data"))
    task_data_dir = det_data_dir / "Task035_KiTS21"

    # setup raw paths
    source_data_dir = task_data_dir / "raw" / "data"
    if not source_data_dir.is_dir():
        raise RuntimeError(
            f"{source_data_dir} should contain the raw data but does not exist."
        )

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(task_data_dir / "prepare.log", level="DEBUG")

    # setup raw splitted dirs
    target_data_dir = task_data_dir / "raw_splitted" / "imagesTr"
    target_data_dir.mkdir(exist_ok=True, parents=True)
    target_label_dir = task_data_dir / "raw_splitted" / "labelsTr"
    target_label_dir.mkdir(exist_ok=True, parents=True)

    # prepare dataset info
    meta = {
        "name": "KiTS21",
        "task": "Task035_KiTS21",
        "target_class": None,
        "test_labels": False,
        "labels": {"0": "tumor", "1": "cyst"},
        "modalities": {"0": "CT"},
        "dim": 3,
    }
    save_json(meta, task_data_dir / "dataset.json")

    # prepare data & label
    case_ids = [p.stem for p in source_data_dir.iterdir() if p.is_dir()]
    case_ids.sort()
    print(f"Found {len(case_ids)} case ids")

    assert len(case_ids) == 300, "Missing cases"

    for cid in maybe_verbose_iterable(case_ids):
        run_prep(
            case_id=cid,
            source_data=source_data_dir,
            target_data_dir=target_data_dir,
            target_label_dir=target_label_dir,
        )


if __name__ == "__main__":
    main()
