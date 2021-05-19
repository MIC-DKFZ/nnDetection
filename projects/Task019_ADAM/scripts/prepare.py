import os
import shutil
from pathlib import Path

from nndet.io import save_json
from nndet.io.prepare import instances_from_segmentation
from nndet.utils.check import env_guard
from nndet.utils.info import maybe_verbose_iterable


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
