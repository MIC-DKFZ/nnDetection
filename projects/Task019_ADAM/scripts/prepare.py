import os
import shutil
from typing import Sequence, Dict, Optional

import SimpleITK as sitk
from pathlib import Path
from loguru import logger

from nndet.io import save_json
from nndet.io.prepare import instances_from_segmentation
from nndet.utils.check import env_guard
from nndet.utils.info import maybe_verbose_iterable
from nndet.utils.clustering import seg_to_instances, remove_classes, reorder_classes


def instances_from_segmentation(
    source_file: Path, output_folder: Path,
    rm_classes: Sequence[int] = None,
    ro_classes: Dict[int, int] = None,
    subtract_one_of_classes: bool = True,
    fg_vs_bg: bool = False,
    file_name: Optional[str] = None
    ):
    """
    1. Optionally removes classes from the segmentation (
    e.g. organ segmentation's which are not useful for detection)

    2. Optionally reorders the segmentation indices

    3. Converts semantic segmentation to instance segmentation's via
    connected components

    Args:
        source_file: path to semantic segmentation file
        output_folder: folder where processed file will be saved
        rm_classes: classes to remove from semantic segmentation
        ro_classes: reorder classes before instances are generated
        subtract_one_of_classes: subtracts one from the classes
            in the instance mapping (detection networks assume
            that classes start from 0)
        fg_vs_bg: map all foreground classes to a single class to run
            foreground vs background detection task.
        file_name: name of saved file (without file type!)
    """
    if subtract_one_of_classes and fg_vs_bg:
        logger.info("subtract_one_of_classes will be ignored because fg_vs_bg is "
                    "active and all foreground classes ill be mapped to 0")

    seg_itk = sitk.ReadImage(str(source_file))
    seg_npy = sitk.GetArrayFromImage(seg_itk)

    if rm_classes is not None:
        seg_npy = remove_classes(seg_npy, rm_classes)

    if ro_classes is not None:
        seg_npy = reorder_classes(seg_npy, ro_classes)

    instances, instance_classes = seg_to_instances(seg_npy)
    if fg_vs_bg:
        num_instances_check = len(instance_classes)
        seg_npy[seg_npy > 0] = 1
        instances, instance_classes = seg_to_instances(seg_npy)
        num_instances = len(instance_classes)
        if num_instances != num_instances_check:
            logger.warning(f"Lost instance: Found {num_instances} instances before "
                           f"fg_vs_bg but {num_instances_check} instances after it")

    if subtract_one_of_classes:
        for key in instance_classes.keys():
            instance_classes[key] -= 1

    if fg_vs_bg:
        for key in instance_classes.keys():
            instance_classes[key] = 0

    seg_itk_new = sitk.GetImageFromArray(instances)
    seg_itk_new.SetSpacing(seg_itk.GetSpacing())
    seg_itk_new.SetOrigin(seg_itk.GetOrigin())
    seg_itk_new.SetDirection(seg_itk.GetDirection())

    if file_name is None:
        suffix_length = sum(map(len, source_file.suffixes))
        file_name = source_file.name[:-suffix_length]

    save_json({"instances": instance_classes}, output_folder / f"{file_name}.json")
    sitk.WriteImage(seg_itk_new, str(output_folder / f"{file_name}.nii.gz"))


def run_prep_fg_v_bg(
        case_id: str,
        source_data: Path,
        target_data_dir,
        target_label_dir: Path,
        struct="pre/struct_aligned.nii.gz",  # bias field corrected and aligned
        tof="pre/TOF.nii.gz",  # tof image
        ):
    struct_path = source_data / case_id / struct
    tof_path = source_data / case_id / tof
    mask_path = source_data / case_id / "aneurysms.nii.gz"

    shutil.copy(struct_path, target_data_dir / f"{case_id}_0000.nii.gz")
    shutil.copy(tof_path, target_data_dir / f"{case_id}_0001.nii.gz")
    instances_from_segmentation(mask_path,
                                target_label_dir,
                                fg_vs_bg=True,
                                file_name=f"{case_id}",
                                )


@env_guard
def main():
    det_data_dir = Path(os.getenv('det_data'))
    task_data_dir = det_data_dir / "Task019FG_ADAM"
    
    # setup raw paths
    source_data_dir = task_data_dir / "raw" / "ADAM_release_subjs"
    if not source_data_dir.is_dir():
        raise RuntimeError(f"{source_data_dir} should contain the raw data but does not exist.")

    # setup raw splitted dirs
    target_data_dir = task_data_dir / "raw_splitted" / "imagesTr"
    target_data_dir.mkdir(exist_ok=True, parents=True)
    target_label_dir = task_data_dir / "raw_splitted" / "labelsTr"
    target_label_dir.mkdir(exist_ok=True, parents=True)

    # prepare dataset info
    meta = {
        "name": "ADAM",
        "task": "Task019FG_ADAM",
        "target_class": None,
        "test_labels": False,
        "labels": {"0": "Aneurysm"}, # since we are running FG vs BG this is not completely correct
        "modalities": {"0": "Structured", "1": "TOF"},
        "dim": 3,
    }
    save_json(meta, task_data_dir / "dataset.json")

    # prepare data
    case_ids = [p.stem for p in source_data_dir.iterdir() if p.is_dir()]
    print(f"Found {len(case_ids)} case ids")
    for cid in maybe_verbose_iterable(case_ids):
        run_prep_fg_v_bg(
            case_id=cid,
            source_data=source_data_dir,
            target_data_dir=target_data_dir,
            target_label_dir=target_label_dir,
            )


if __name__ == "__main__":
    main()
