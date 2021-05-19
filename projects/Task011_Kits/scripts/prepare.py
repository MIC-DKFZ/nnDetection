import shutil
import os
import sys
from pathlib import Path

from loguru import logger

from nndet.io import save_json
from nndet.io.prepare import create_test_split
from nndet.utils.check import env_guard
from nndet.utils.info import maybe_verbose_iterable


@env_guard
def main():
    det_data_dir = Path(os.getenv('det_data'))
    task_data_dir = det_data_dir / "Task011_Kits"
    source_data_dir = task_data_dir / "raw"

    if not source_data_dir.is_dir():
        raise RuntimeError(f"{source_data_dir} should contain the raw data but does not exist.")

    splitted_dir = task_data_dir / "raw_splitted"
    target_data_dir = task_data_dir / "raw_splitted" / "imagesTr"
    target_data_dir.mkdir(exist_ok=True, parents=True)
    target_label_dir = task_data_dir / "raw_splitted" / "labelsTr"
    target_label_dir.mkdir(exist_ok=True, parents=True)

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(task_data_dir / "prepare.log", level="DEBUG")

    # save meta info
    dataset_info = {
        "name": "Kits",
        "task": "Task011_Kits",
        "target_class": None,
        "test_labels": True,

        "seg2det_stuff": [1,], # define stuff classes: kidney
        "seg2det_things": [2,], # define things classes: tumor
        "min_size": 3.,

        "labels": {"0": "lesion"},
        "labels_stuff": {"1": "kidney"},
        "modalities": {"0": "CT"},
        "dim": 3,
    }
    save_json(dataset_info, task_data_dir / "dataset.json")

    # prepare cases
    cases = [str(c.name) for c in source_data_dir.iterdir() if c.is_dir()]
    for c in maybe_verbose_iterable(cases):
        logger.info(f"Copy case {c}")
        case_id = int(c.split("_")[-1])
        if case_id < 210:
            shutil.copy(source_data_dir / c / "imaging.nii.gz", target_data_dir / f"{c}_0000.nii.gz")
            shutil.copy(source_data_dir / c / "segmentation.nii.gz", target_label_dir / f"{c}.nii.gz")

    # create an artificial test split
    create_test_split(splitted_dir=splitted_dir,
                      num_modalities=1,
                      test_size=0.3,
                      random_state=0,
                      shuffle=True,
                      )


if __name__ == '__main__':
    main()
