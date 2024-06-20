import concurrent.futures
import os
import sys
from itertools import repeat
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from loguru import logger
from sklearn.model_selection import StratifiedKFold

from nndet.io import save_json
from nndet.io.load import load_json, save_pickle
from nndet.utils.check import env_guard

PN9CLASSES = {
    "CN": "0",
    "0-3SN": "1",
    "3-10SN": "2",
    "10-30SN": "3",
    ">30SN": "4",
    "0-5PSN": "5",
    ">5PSN": "6",
    "0-5GGN": "7",
    ">5GGN": "8",
}


def add_ellipsoid(
    mask: np.ndarray,
    idx: int,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    z_min: int,
    z_max: int,
) -> Tuple[np.ndarray, Dict[str, bool]]:
    # will overwrite existing values in mask!
    assert idx > 0

    dx2 = (x_max - x_min) / 2.0
    cx = (x_min + x_max) / 2.0 - 1  # one indexed
    assert dx2 >= 0

    dy2 = (y_max - y_min) / 2.0
    cy = (y_min + y_max) / 2.0 - 1  # one indexed
    assert dy2 >= 0

    dz2 = max((z_max - z_min) / 2.0, 1.0)
    cz = (z_min + z_max) / 2.0 - 1.0 - 1  # one indexed
    assert dz2 >= 0

    object_flags = {
        "added": False,
        "overlap": False,
        "outside": False,
    }
    for x in range(int(x_min) - 2, int(x_max)):
        for y in range(int(y_min) - 2, int(y_max)):
            for z in range(int(z_min) - 2, int(z_max)):
                if (x - cx) ** 2 / dx2**2 + (y - cy) ** 2 / dy2**2 + (z - cz) ** 2 / dz2**2 <= 1.0:

                    # coordinate outside of scan
                    if x >= mask.shape[2] or y >= mask.shape[1] or z >= mask.shape[0]:
                        object_flags["outside"] = True
                        continue

                    # coordiante already occupied
                    if mask[z, y, x] > 0:
                        object_flags["overlap"] = True

                    # add object
                    mask[z, y, x] = idx
                    object_flags["added"] = True
    return mask, object_flags


def run_prep(
    case_id: str,
    labels_df: pd.DataFrame,
    source_data: Path,
    target_data_dir: Path,
    target_label_dir: Path,
) -> str:
    logger.info(f"Processing case {case_id}")
    data_np = np.load(source_data / f"{case_id}_zoom.npy")
    assert data_np.ndim == 4
    mask_np = np.zeros_like(data_np[0], dtype=np.int32)

    boxes_image_df = labels_df[labels_df["pid"] == int(case_id)]
    # sort objects from largest to smalles -> smaller objects will only partially cover larger objects
    # so we don't loose objects when creating the masks
    boxes_image_df = boxes_image_df.sort_values("volume", ascending=False)

    # add objects
    other_info = {}
    instances = {}
    idx = 1
    object_flags = []
    for _, box_row in boxes_image_df.iterrows():
        instances[str(idx)] = int(PN9CLASSES[box_row["nodule_class"]])
        other_info[str(idx)] = {
            "nodule_id": box_row["nodule_id"],
            "nodule_class": box_row["nodule_class"],
            "volume": box_row["volume"],
        }

        # insert elipsoid
        x_min = float(box_row["xmin"])
        x_max = float(box_row["xmax"])
        y_min = float(box_row["ymin"])
        y_max = float(box_row["ymax"])
        z_min = float(box_row["zmin"])
        z_max = float(box_row["zmax"])
        mask_np, _object_flags = add_ellipsoid(
            mask_np,
            idx=idx,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
        )
        object_flags.append(_object_flags)
        idx += 1

    # add logging message
    objects_added = [obj["added"] for obj in object_flags]
    all_objects_added = all(objects_added)
    objects_overlap = [obj["overlap"] for obj in object_flags]
    any_objects_overlap = any(objects_overlap)
    objects_outside = [obj["outside"] for obj in object_flags]
    any_objects_outside = any(objects_outside)

    if not all_objects_added:
        logger.error(f"Not all objects were added for {case_id}: {objects_added} (array starting from 1 to N)!!!")

    _log_str = ""
    if any_objects_overlap:
        _log_str += f"Object Overlap {objects_overlap}"
    if any_objects_outside:
        _log_str += f"Object Outside {objects_outside}"
    if _log_str:
        _log_str = f"Found special information in case {case_id}: {_log_str} (arrays starting from 1 to N)"
        logger.warning(_log_str)

    # save prepared case
    data_itk = sitk.GetImageFromArray(data_np[0].astype(np.float32))
    sitk.WriteImage(data_itk, str(target_data_dir / f"{case_id}_0000.nii.gz"))

    mask_itk = sitk.GetImageFromArray(mask_np.astype(np.int32))
    sitk.WriteImage(mask_itk, str(target_label_dir / f"{case_id}.nii.gz"))
    instances_nodule = {k: 0 for k in instances.keys()}
    save_json(
        {"instances": instances_nodule, "instances_cls": instances, "meta": other_info},
        target_label_dir / f"{case_id}.json",
    )

    return case_id


def create_custom_split(
    case_ids: Sequence[str],
    label_dir: Path,
) -> List[Dict[str, List[str]]]:
    logger.info(f"Received {len(case_ids)} case ids \n{case_ids}")
    patient_ids = None
    # derive class info
    case_classes = []
    all_classes = []
    for cid in case_ids:
        case_instances = load_json(label_dir / f"{cid}.json")
        case_instances = [int(i) for i in case_instances["instances"].values()]

        case_classes.append(case_instances)
        all_classes.extend(case_instances)

    _, class_counts = np.unique(all_classes, return_counts=True)
    logger.info(f"Class count: {class_counts}")

    reduced_classes = []
    for cc in case_classes:
        if len(cc) == 0:
            reduced_classes.append(-1)
        else:
            rarest_class_index = np.argmin([class_counts[_cc] for _cc in cc])
            reduced_classes.append(cc[rarest_class_index])

    # create stratified group k fold
    splits = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(case_ids, reduced_classes, patient_ids)):
        train_cids = [case_ids[_i] for _i in train_idx]
        val_cids = [case_ids[_i] for _i in val_idx]
        intersection_cids = set(train_cids).intersection(val_cids)

        train_reduced_classes = [reduced_classes[_i] for _i in train_idx]
        val_reduced_classes = [reduced_classes[_i] for _i in val_idx]
        train_reduced_classes = {k: i for k, i in zip(*np.unique(train_reduced_classes, return_counts=True))}
        val_reduced_classes = {k: i for k, i in zip(*np.unique(val_reduced_classes, return_counts=True))}

        assert not intersection_cids
        logger.info(
            f"Generated fold {fold_idx} with {len(train_cids)} "
            f"train {len(val_cids)} val cases. "
            f"Intersection {intersection_cids} (should be empty)."
            f"Reduced classes: train {train_reduced_classes} val {val_reduced_classes}"
        )

        splits.append({"train": train_cids, "val": val_cids})
    return splits


@env_guard
def main():
    det_data_dir = Path(os.getenv("det_data"))
    task_data_dir = det_data_dir / "Task053_PN9"

    # setup raw paths
    raw_data_dir = task_data_dir / "raw"
    source_data_dir_train = raw_data_dir / "train"
    if not source_data_dir_train.is_dir():
        raise RuntimeError(f"{source_data_dir_train} should contain the raw data but does not exist.")

    source_data_dir_test = raw_data_dir / "test"
    if not source_data_dir_test.is_dir():
        raise RuntimeError(f"{source_data_dir_test} should contain the raw data but does not exist.")

    source_labels_train = raw_data_dir / "train_anno.csv"
    if not source_labels_train.is_file():
        raise RuntimeError(f"{source_labels_train} should contain the train annotations.")

    train_labels_df = pd.read_csv(source_labels_train)
    train_labels_df["volume"] = train_labels_df.apply(
        lambda row: float(row["xmax"] - row["xmin"])
        * float(row["ymax"] - row["ymin"])
        * float(row["zmax"] - row["zmin"]),
        axis=1,
    )

    # setup raw splitted dirs
    target_data_dir_train = task_data_dir / "raw_splitted" / "imagesTr"
    target_data_dir_train.mkdir(exist_ok=True, parents=True)
    target_label_dir_train = task_data_dir / "raw_splitted" / "labelsTr"
    target_label_dir_train.mkdir(exist_ok=True, parents=True)

    target_data_dir_test = task_data_dir / "raw_splitted" / "imagesTs"
    target_data_dir_test.mkdir(exist_ok=True, parents=True)
    target_label_dir_test = task_data_dir / "raw_splitted" / "labelsTs"
    target_label_dir_test.mkdir(exist_ok=True, parents=True)

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(task_data_dir / "prepare.log", level="DEBUG")

    # prepare dataset info
    meta = {
        "dim": 3,
        "task": "Task053_PN9",
        "target_class": None,
        "test_labels": True,
        # "labels": {item: key for key, item in PN9CLASSES.items()},
        "labels": {"0": "nodule"},
        # scans are already preprocessed, default CT scheme does not apply
        "modalities": {"0": "CT_preprocessed"},
    }
    save_json(meta, task_data_dir / "dataset.json")

    ############################### TRAIN DATA #################################
    # prepare data & label
    logger.info("Prepare Train Data")
    case_ids = sorted([(p.stem).rsplit("_", 1)[0] for p in source_data_dir_train.glob("*.npy")])

    # preprocess data
    logger.info(f"Found {len(case_ids)} case ids")
    assert len(set(case_ids)) == (6037 + 670), "Missing cases in train data"

    num_processes = int(os.getenv("det_num_threads", 4))
    if num_processes < 1:
        logger.info("Running in single process mode")
        for cid in case_ids:
            run_prep(
                case_id=cid,
                labels_df=train_labels_df,
                source_data=source_data_dir_train,
                target_data_dir=target_data_dir_train,
                target_label_dir=target_label_dir_train,
            )
    else:
        logger.info(f"Running in multi process mode with {num_processes} processes")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            for cid in executor.map(
                run_prep,
                case_ids,
                repeat(train_labels_df),
                repeat(source_data_dir_train),
                repeat(target_data_dir_train),
                repeat(target_label_dir_train),
            ):
                logger.info(f"Finished processing case {cid}")

    ############################### Splits #####################################
    # create custom split
    logger.info("Create Custom Split for PN9")
    train_ids_path = raw_data_dir / "train.txt"
    if not train_ids_path.is_file():
        raise RuntimeError(f"File {train_ids_path} does not exist.")
    val_ids_path = raw_data_dir / "val.txt"
    if not val_ids_path.is_file():
        raise RuntimeError(f"File {val_ids_path} does not exist.")

    train_case_ids = open(train_ids_path).read().split()
    val_case_ids = open(val_ids_path).read().split()

    # single split
    single_split = [{"train": train_case_ids, "val": val_case_ids}]
    save_json(single_split, task_data_dir / "splits_single.json")
    save_pickle(single_split, task_data_dir / "splits_single.pkl")

    # official val splits
    splits_train = create_custom_split(
        case_ids=train_case_ids,
        label_dir=target_label_dir_train,
    )
    fixed_val_splits = []
    for split in splits_train:
        fixed_val_splits.append({"train": split["train"], "val": split["val"], "val_off": val_case_ids})
    save_json(fixed_val_splits, task_data_dir / "splits_val.json")
    save_pickle(fixed_val_splits, task_data_dir / "splits_val.pkl")

    # all (train + val) data 5 Fold cv
    splits_all = create_custom_split(
        case_ids=case_ids,
        label_dir=target_label_dir_train,
    )
    save_json(splits_all, task_data_dir / "splits_all.json")
    save_pickle(splits_all, task_data_dir / "splits_all.pkl")

    ############################### TEST DATA ##################################
    # prepare test data
    logger.info("Prepare Test Data")
    test_case_ids = sorted([(p.stem).rsplit("_", 1)[0] for p in source_data_dir_test.glob("*.npy")])

    source_labels_test = raw_data_dir / "test_anno.csv"
    if not source_labels_test.is_file():
        raise RuntimeError(f"{source_labels_test} should contain the train annotations.")
    test_labels_df = pd.read_csv(source_labels_test)
    test_labels_df["volume"] = test_labels_df.apply(
        lambda row: float(row["xmax"] - row["xmin"])
        * float(row["ymax"] - row["ymin"])
        * float(row["zmax"] - row["zmin"]),
        axis=1,
    )

    logger.info(f"Found {len(test_case_ids)} case ids for testing")
    assert len(set(test_case_ids)) == 2091, "Missing cases in test data"

    num_processes = int(os.getenv("det_num_threads", 4))
    if num_processes < 1:
        logger.info("Running in single process mode")
        for cid in test_case_ids:
            run_prep(
                case_id=cid,
                labels_df=test_labels_df,
                source_data=source_data_dir_test,
                target_data_dir=target_data_dir_test,
                target_label_dir=target_label_dir_test,
            )
    else:
        logger.info(f"Running in multi process mode with {num_processes} processes")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            for cid in executor.map(
                run_prep,
                test_case_ids,
                repeat(test_labels_df),
                repeat(source_data_dir_test),
                repeat(target_data_dir_test),
                repeat(target_label_dir_test),
            ):
                logger.info(f"Finished processing case {cid}")


if __name__ == "__main__":
    main()
