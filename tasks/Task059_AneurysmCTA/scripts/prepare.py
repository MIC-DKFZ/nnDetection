import concurrent
import os
import shutil
import sys
from itertools import repeat
from pathlib import Path
from typing import List

import cc3d
import numpy as np
import pandas as pd
import SimpleITK as sitk
from loguru import logger
from sklearn.cluster import SpectralClustering

from nndet.io.load import save_json
from nndet.utils.check import env_guard
from nndet.utils.info import maybe_verbose_iterable

IS_RUPTURED = {"yes": True, "no": False}


def prepare_case(
    case_id: str,
    source_data_dir: Path,
    target_data_dir: Path,
    source_label_dir: Path,
    target_label_dir: Path,
    labels_df: pd.DataFrame,
) -> None:
    logger.info(f"Processing case {case_id}")
    source_image_path = source_data_dir / f"{case_id}.nii.gz"
    if not source_image_path.is_file():
        raise RuntimeError(f"Image file {source_image_path} does not exist.")
    source_label_path = source_label_dir / f"{case_id}.nii.gz"
    if not source_label_path.is_file():
        raise RuntimeError(f"Label file {source_label_path} does not exist.")

    case_labels = labels_df.loc[case_id]

    # load label informaton
    label_itk = sitk.ReadImage(str(source_label_path))
    label_data = sitk.GetArrayFromImage(label_itk)

    # Run connected components on labels via cc3d
    labels_out, num_instances = cc3d.connected_components(label_data, connectivity=6, return_N=True)

    # Cross check labels with label dataframe
    num_aneurysms = case_labels["num_IAs"]
    if num_instances != num_aneurysms:
        logger.info(
            f"CCN: Number of aneurysms in {case_id} does not match: {num_aneurysms} (CC) vs {num_aneurysms} (CSV). "
        )
        if num_instances < num_aneurysms:
            points = np.array(np.nonzero(labels_out)).T
            clustering = SpectralClustering(n_clusters=num_aneurysms, assign_labels="discretize", random_state=0).fit(
                points
            )
            for instance_index in range(num_aneurysms):
                instance_points = points[clustering.labels_ == instance_index]
                labels_out[instance_points[:, 0], instance_points[:, 1], instance_points[:, 2]] = instance_index + 1
            logger.info(
                f"CCN {case_id}: Running in spectral clustering to bump up the number of instances to {num_aneurysms}"
            )
        else:
            vols = [np.sum(labels_out == instance_index) for instance_index in range(1, num_instances + 1)]
            tmp_labels_out = np.zeros_like(labels_out)
            for new_instance_index, old_instance_index in enumerate(
                np.argsort(vols)[::-1][:num_aneurysms] + 1, start=1
            ):
                tmp_labels_out[labels_out == old_instance_index] = new_instance_index
            labels_out = tmp_labels_out
            logger.info(f"CCN {case_id}: Removing min volumes to bring down the number of instances to {num_aneurysms}")

    new_label_itk = sitk.GetImageFromArray(labels_out)
    new_label_itk.CopyInformation(label_itk)

    # Copy data to target dir
    shutil.copy2(source_data_dir / f"{case_id}.nii.gz", target_data_dir / f"{case_id}_0000.nii.gz")

    # Save labels to target dir
    sitk.WriteImage(new_label_itk, str(target_label_dir / f"{case_id}.nii.gz"))

    # Create and save meta information
    meta_info = {
        "instances": {str(int(i)): 0 for i in np.unique(labels_out) if i != 0},
        "num_aneurysms_cc": int(num_aneurysms),
        "subset": str(case_labels["subset"]),
        "institution_id": int(case_labels["institution_id"]),
        "age": int(case_labels["age"]),
        "gender": str(case_labels["gender"]),
        "is_ruptured": IS_RUPTURED[case_labels["is_ruptured"]],
        "num_IAs": int(case_labels["num_IAs"]),
    }
    save_json(meta_info, target_label_dir / f"{case_id}.json")
    return case_id


def process_dataset(
    case_ids: List[str],
    source_data_dir: Path,
    target_data_dir: Path,
    source_label_dir: Path,
    target_label_dir: Path,
    labels_df: pd.DataFrame,
):
    num_processes = int(os.getenv("det_num_threads", 4))
    if num_processes < 1:
        logger.info("Running in single process mode")
        for cid in maybe_verbose_iterable(case_ids):
            prepare_case(
                case_id=cid,
                source_data_dir=source_data_dir,
                target_data_dir=target_data_dir,
                source_label_dir=source_label_dir,
                target_label_dir=target_label_dir,
                labels_df=labels_df,
            )
            logger.info(f"Finished processing case {cid}")
    else:
        logger.info(f"Running in multi process mode with {num_processes} processes")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            for cid in executor.map(
                prepare_case,
                case_ids,
                repeat(source_data_dir),
                repeat(target_data_dir),
                repeat(source_label_dir),
                repeat(target_label_dir),
                repeat(labels_df),
            ):
                logger.info(f"Finished processing case {cid}")


@env_guard
def main():
    task = "Task059_AneurysmCTA"
    det_data_dir = Path(os.getenv("det_data"))
    task_data_dir = det_data_dir / task
    assert task_data_dir.is_dir(), f"{task_data_dir} does not exist."

    # traning data paths
    source_data_dir = task_data_dir / "raw" / "imagesTr"
    source_label_dir = task_data_dir / "raw" / "labelsTr"
    internal_csv_path = task_data_dir / "raw" / "internal_instances.csv"
    if not source_data_dir.is_dir():
        raise RuntimeError(f"{source_data_dir} should contain the raw data but does not exist.")
    if not source_label_dir.is_dir():
        raise RuntimeError(f"{source_label_dir} should contain the raw labels but does not exist.")
    if not internal_csv_path.is_file():
        raise RuntimeError(
            f"{internal_csv_path} should contain the csv labels with the number of aneurysms but does not exist."
        )

    # internal testing data paths
    source_data_test_int_dir = task_data_dir / "raw" / "imagesTs_internal"
    source_label_test_int_dir = task_data_dir / "raw" / "labelsTs_internal"
    if not source_data_test_int_dir.is_dir():
        raise RuntimeError(f"{source_data_test_int_dir} should contain the raw data but does not exist.")
    if not source_label_test_int_dir.is_dir():
        raise RuntimeError(f"{source_label_test_int_dir} should contain the raw labels but does not exist.")

    # external testing data paths
    source_data_test_ext_dir = task_data_dir / "raw" / "imagesTs_external"
    source_label_test_ext_dir = task_data_dir / "raw" / "labelsTs_external"
    external_csv_path = task_data_dir / "raw" / "external_instances.csv"
    if not source_data_test_ext_dir.is_dir():
        raise RuntimeError(f"{source_data_test_ext_dir} should contain the raw data but does not exist.")
    if not source_label_test_ext_dir.is_dir():
        raise RuntimeError(f"{source_label_test_ext_dir} should contain the raw labels but does not exist.")
    if not external_csv_path.is_file():
        raise RuntimeError(
            f"{external_csv_path} should contain the csv labels with the number of aneurysms but does not exist."
        )

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(task_data_dir / "prepare2.log", level="DEBUG")

    # setup raw splitted dirs
    raw_splitted_dir = task_data_dir / "raw_splitted"
    target_data_dir = raw_splitted_dir / "imagesTr"
    target_data_dir.mkdir(exist_ok=True, parents=True)
    target_label_dir = raw_splitted_dir / "labelsTr"
    target_label_dir.mkdir(exist_ok=True, parents=True)

    test_int_target_data_dir = raw_splitted_dir / "imagesTs_internal"
    test_int_target_data_dir.mkdir(exist_ok=True, parents=True)
    test_int_target_label_dir = raw_splitted_dir / "labelsTs_internal"
    test_int_target_label_dir.mkdir(exist_ok=True, parents=True)

    test_ext_target_data_dir = raw_splitted_dir / "imagesTs_external"
    test_ext_target_data_dir.mkdir(exist_ok=True, parents=True)
    test_ext_target_label_dir = raw_splitted_dir / "labelsTs_external"
    test_ext_target_label_dir.mkdir(exist_ok=True, parents=True)

    # load meta information for labels
    internal_labels_df = pd.read_csv(internal_csv_path).set_index("instance_id")
    external_labels_df = pd.read_csv(external_csv_path).set_index("instance_id")

    # prepare dataset info
    meta = {
        "task": task,
        "target_class": None,
        "test_labels": True,
        "labels": {"0": "aneurysm"},
        "modalities": {"0": "CT"},
        "dim": 3,
        "annotation_style": "seg",
    }
    save_json(meta, task_data_dir / "dataset.json")

    # prepare training data
    train_case_ids = [p.name.rsplit(".", 2)[0] for p in source_data_dir.iterdir() if p.name.endswith(".nii.gz")]
    train_case_ids.sort()
    logger.info(f"Found {len(train_case_ids)} case ids")
    assert len(train_case_ids) == 1186, "Missing cases"
    process_dataset(
        case_ids=train_case_ids,
        source_data_dir=source_data_dir,
        target_data_dir=target_data_dir,
        source_label_dir=source_label_dir,
        target_label_dir=target_label_dir,
        labels_df=internal_labels_df,
    )

    # prepare internal testing data
    internal_test_case_ids = [
        p.name.rsplit(".", 2)[0] for p in source_data_test_int_dir.iterdir() if p.name.endswith(".nii.gz")
    ]
    internal_test_case_ids.sort()
    logger.info(f"Found {len(internal_test_case_ids)} case ids")
    assert len(internal_test_case_ids) == 152, "Missing cases"
    process_dataset(
        case_ids=internal_test_case_ids,
        source_data_dir=source_data_test_int_dir,
        target_data_dir=test_int_target_data_dir,
        source_label_dir=source_label_test_int_dir,
        target_label_dir=test_int_target_label_dir,
        labels_df=internal_labels_df,
    )

    # prepare external testing data
    external_test_case_ids = [
        p.name.rsplit(".", 2)[0] for p in source_data_test_ext_dir.iterdir() if p.name.endswith(".nii.gz")
    ]
    external_test_case_ids.sort()
    logger.info(f"Found {len(external_test_case_ids)} case ids")
    assert len(external_test_case_ids) == 138, "Missing cases"
    process_dataset(
        case_ids=external_test_case_ids,
        source_data_dir=source_data_test_ext_dir,
        target_data_dir=test_ext_target_data_dir,
        source_label_dir=source_label_test_ext_dir,
        target_label_dir=test_ext_target_label_dir,
        labels_df=external_labels_df,
    )


if __name__ == "__main__":
    main()
