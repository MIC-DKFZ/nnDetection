import argparse
import os
import shutil
import sys
from itertools import repeat
from multiprocessing import Pool, Value
from pathlib import Path

from loguru import logger
from nndet.io.load import save_json

from nndet.io.prepare import maybe_split_4d_nifti, create_test_split

from nndet.io import get_case_ids_from_dir, load_json, save_yaml
from nndet.utils.check import env_guard
from nndet.utils.info import maybe_verbose_iterable


def process_case(case_id,
                 source_images,
                 source_labels,
                 target_images,
                 target_labels,
                 ):
    logger.info(f"Processing case {case_id}")
    maybe_split_4d_nifti(source_images / f"{case_id}.nii.gz", target_images)
    shutil.copy2(source_labels / f"{case_id}.nii.gz", target_labels)


@env_guard
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tasks', type=str, nargs='+',
                        help="One or multiple of: Task003_Liver, Task007_Pancreas, "
                        "Task008_HepaticVessel, Task010_Colon",
                        )
    args = parser.parse_args()
    tasks = args.tasks

    decathlon_props = {
        "Task003_Liver": {
            "seg2det_stuff": [1, ],  # liver
            "seg2det_things": [2, ],  # cancer
            "min_size": 3.,
            "labels": {"0": "cancer"},
            "labels_stuff": {"1": "liver"},
        },
        "Task007_Pancreas": {
            "seg2det_stuff": [1, ],  # pancreas
            "seg2det_things": [2, ],
            "min_size": 3.,
            "labels": {"0": "cancer"},
            "labels_stuff": {"1": "pancreas"},
        },
        "Task008_HepaticVessel": {
            "seg2det_stuff": [1, ],  # vessel
            "seg2det_things": [2, ],
            "min_size": 3.,
            "labels": {"0": "tumour"},
            "labels_stuff": {"1": "vessel"},
        },
        "Task010_Colon": {
            "seg2det_stuff": [],
            "seg2det_things": [1, ],
            "min_size": 3.,
            "labels": {"0": "cancer"},
            "labels_stuff": {},
        },
    }

    basedir = Path(os.getenv('det_data'))
    for task in tasks:
        task_data_dir = basedir / task

        logger.remove()
        logger.add(sys.stdout, level="INFO")
        logger.add(task_data_dir / "prepare.log", level="DEBUG")
        logger.info(f"Preparing task: {task}")

        source_raw_dir = task_data_dir / "raw"
        source_data_dir = source_raw_dir / "imagesTr"
        source_labels_dir = source_raw_dir / "labelsTr"
        splitted_dir = task_data_dir / "raw_splitted"

        if not source_data_dir.is_dir():
            raise ValueError(f"Exptected training images at {source_data_dir}")
        if not source_labels_dir.is_dir():
            raise ValueError(f"Exptected training labels at {source_labels_dir}")
        if not (p := source_raw_dir / "dataset.json").is_file():
            raise ValueError(f"Expected dataset json to be located at {p}")

        target_data_dir = splitted_dir / "imagesTr"
        target_label_dir = splitted_dir / "labelsTr"
        target_data_dir.mkdir(parents=True, exist_ok=True)
        target_label_dir.mkdir(parents=True, exist_ok=True)

        # preapre meta
        original_meta = load_json(source_raw_dir / "dataset.json")

        dataset_info = {
            "task": task,
            "name": original_meta["name"],
            
            "target_class": None,
            "test_labels": True,

            "modalities": original_meta["modality"],
            "dim": 3,
            "info": {
                "original_labels": original_meta["labels"],
                "original_numTraining": original_meta["numTraining"],
            },
        }
        dataset_info.update(decathlon_props[task])
        save_json(dataset_info, task_data_dir / "dataset.json")

        # prepare data and labels
        case_ids = get_case_ids_from_dir(source_data_dir, remove_modality=False)
        case_ids = sorted([c for c in case_ids if c])
        logger.info(f"Found {len(case_ids)} for preparation.")

        for cid in maybe_verbose_iterable(case_ids):
            process_case(cid,
                         source_data_dir,
                         source_labels_dir,
                         target_data_dir,
                         target_label_dir,
                         )

        # with Pool(processes=6) as p:
        #     p.starmap(process_case, zip(case_ids,
        #                                 repeat(source_images),
        #                                 repeat(source_labels),
        #                                 repeat(target_images),
        #                                 repeat(target_labels),
        #                                 ))

        # create an artificial test split
        create_test_split(splitted_dir=splitted_dir,
                          num_modalities=1,
                          test_size=0.3,
                          random_state=0,
                          shuffle=True,
                          )


if __name__ == '__main__':
    main()
