import os
import shutil
from pathlib import Path

import SimpleITK as sitk

from nndet.io import save_json
from nndet.utils.check import env_guard
from nndet.utils.info import maybe_verbose_iterable


def run_prep(source_data: Path, source_label: Path,
             target_data_dir, target_label_dir: Path):
    case_id = f"{(source_data.stem).rsplit('_', 1)[0]}"

    shutil.copy(source_data, target_data_dir / f"{case_id}_0000.nii.gz")
    shutil.copy(source_label, target_label_dir / f"{case_id}.nii.gz")  # rename label file to match data
    label_itk = sitk.ReadImage(str(source_label))
    
    label_np = sitk.GetArrayFromImage(label_itk)
    instances = {int(_id + 1): 0 for _id in range(label_np.max())}
    save_json({"instances": instances}, target_label_dir / f"{case_id}")


@env_guard
def main():
    det_data_dir = Path(os.getenv('det_data'))
    task_data_dir = det_data_dir / "Task017_CADA"
    
    # setup raw paths
    source_data_dir = task_data_dir / "raw" / "train_dataset"
    if not source_data_dir.is_dir():
        raise RuntimeError(f"{source_data_dir} should contain the raw data but does not exist.")
    source_label_dir = task_data_dir / "raw" / "train_mask_images"
    if not source_label_dir.is_dir():
        raise RuntimeError(f"{source_label_dir} should contain the raw labels but does not exist.")

    # setup raw splitted dirs
    target_data_dir = task_data_dir / "raw_splitted" / "imagesTr"
    target_data_dir.mkdir(exist_ok=True, parents=True)
    target_label_dir = task_data_dir / "raw_splitted" / "labelsTr"
    target_label_dir.mkdir(exist_ok=True, parents=True)

    # prepare dataset info
    meta = {
        "name": "CADA",
        "task": "Task017_CADA",
        
        "target_class": None,
        "test_labels": False,
        
        "labels": {"0": "aneurysm"},
        "modalities": {"0": "CT"},
        "dim": 3,
    }
    save_json(meta, task_data_dir / "dataset.json")

    # prepare data & label
    case_ids = [(p.stem).rsplit('_', 1)[0] for p in source_data_dir.glob("*.nii.gz")]
    print(f"Found {len(case_ids)} case ids")
    for cid in maybe_verbose_iterable(case_ids):
        run_prep(
            source_data=source_data_dir / f"{cid}_orig.nii.gz",
            source_label=source_label_dir / f"{cid}_labeledMasks.nii.gz",
            target_data_dir=target_data_dir,
            target_label_dir=target_label_dir,
            )


if __name__ == "__main__":
    main()
