# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from nndet.io.paths import get_task
from nndet.utils.check import env_guard


@env_guard
def boxes2nii():
    """
    Only for visualisation purposes.
    """
    import argparse
    import os
    from pathlib import Path

    import numpy as np
    import SimpleITK as sitk
    from loguru import logger

    from nndet.io import load_pickle, save_json
    from nndet.io.paths import get_task, get_training_dir
    from nndet.utils.info import maybe_verbose_iterable

    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument("model", type=str, help="model name, e.g. RetinaUNetV0")
    parser.add_argument("fold", type=int, help="experiment fold")
    parser.add_argument(
        "-o",
        "--overwrites",
        type=str,
        nargs="+",
        help="overwrites for config file",
        required=False,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Minimum probability of predictions",
        required=False,
        default=0.5,
    )
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    model = args.model
    fold = args.fold
    task = args.task
    overwrites = args.overwrites
    test = args.test
    threshold = args.threshold

    task_name = get_task(task, name=True, models=True)
    task_dir = Path(os.getenv("det_models")) / task_name

    training_dir = get_training_dir(task_dir / model, fold)

    overwrites = overwrites if overwrites is not None else []
    overwrites.append("host.parent_data=${env:det_data}")
    overwrites.append("host.parent_results=${env:det_models}")

    prediction_dir = (
        training_dir / "test_predictions" if test else training_dir / "val_predictions"
    )
    save_dir = (
        training_dir / "test_predictions_nii"
        if test
        else training_dir / "val_predictions_nii"
    )
    save_dir.mkdir(exist_ok=True)

    case_ids = [p.stem.rsplit("_", 1)[0] for p in prediction_dir.glob("*_boxes.pkl")]
    case_ids.sort()
    for cid in maybe_verbose_iterable(case_ids):
        res = load_pickle(prediction_dir / f"{cid}_boxes.pkl")

        instance_mask = np.zeros(res["original_size_of_raw_data"], dtype=np.uint8)

        boxes = res["pred_boxes"]
        scores = res["pred_scores"]
        labels = res["pred_labels"]

        _mask = scores >= threshold
        boxes = boxes[_mask]
        labels = labels[_mask]
        scores = scores[_mask]

        idx = np.argsort(scores)
        scores = scores[idx]
        boxes = boxes[idx]
        labels = labels[idx]

        prediction_meta = {}
        for instance_id, (pbox, pscore, plabel) in enumerate(
            zip(boxes, scores, labels), start=1
        ):
            mask_slicing = [
                slice(int(pbox[0]) + 1, int(pbox[2])),
                slice(int(pbox[1]) + 1, int(pbox[3])),
            ]
            if instance_mask.ndim == 3:
                mask_slicing.append(slice(int(pbox[4]) + 1, int(pbox[5])))
            instance_mask[tuple(mask_slicing)] = instance_id

            prediction_meta[int(instance_id)] = {
                "score": float(pscore),
                "label": int(plabel),
                "box": list(map(int, pbox)),
            }

        logger.info(
            f"Created instance mask {cid} with {instance_mask.max()} instances."
        )

        instance_mask_itk = sitk.GetImageFromArray(instance_mask)
        instance_mask_itk.SetOrigin(res["itk_origin"])
        instance_mask_itk.SetDirection(res["itk_direction"])
        instance_mask_itk.SetSpacing(res["itk_spacing"])

        sitk.WriteImage(instance_mask_itk, str(save_dir / f"{cid}_boxes.nii.gz"))
        save_json(prediction_meta, save_dir / f"{cid}_boxes.json")


@env_guard
def boxes2nii2():
    """
    Only for visualisation purposes.
    Creates binary mask and puts each object into a separate channel
    """
    import argparse
    import os
    from pathlib import Path

    import numpy as np
    import SimpleITK as sitk
    from loguru import logger

    from nndet.io import load_pickle, save_json
    from nndet.io.paths import get_task, get_training_dir
    from nndet.utils.info import maybe_verbose_iterable

    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument("model", type=str, help="model name, e.g. RetinaUNetV0")
    parser.add_argument("fold", type=int, help="experiment fold")
    parser.add_argument(
        "-o",
        "--overwrites",
        type=str,
        nargs="+",
        help="overwrites for config file",
        required=False,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Minimum probability of predictions",
        required=False,
        default=0.5,
    )
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    model = args.model
    fold = args.fold
    task = args.task
    overwrites = args.overwrites
    test = args.test
    threshold = args.threshold

    task_name = get_task(task, name=True, models=True)
    task_dir = Path(os.getenv("det_models")) / task_name

    training_dir = get_training_dir(task_dir / model, fold)

    overwrites = overwrites if overwrites is not None else []
    overwrites.append("host.parent_data=${env:det_data}")
    overwrites.append("host.parent_results=${env:det_models}")

    prediction_dir = (
        training_dir / "test_predictions" if test else training_dir / "val_predictions"
    )
    save_dir = (
        training_dir / "test_predictions_nii2"
        if test
        else training_dir / "val_predictions_nii2"
    )
    save_dir.mkdir(exist_ok=True)

    case_ids = [p.stem.rsplit("_", 1)[0] for p in prediction_dir.glob("*_boxes.pkl")]
    for cid in maybe_verbose_iterable(case_ids):
        res = load_pickle(prediction_dir / f"{cid}_boxes.pkl")

        boxes = res["pred_boxes"]
        scores = res["pred_scores"]
        labels = res["pred_labels"]

        _mask = scores >= threshold
        boxes = boxes[_mask]
        labels = labels[_mask]
        scores = scores[_mask]

        idx = np.argsort(scores)
        scores = scores[idx]
        boxes = boxes[idx]
        labels = labels[idx]

        prediction_meta = {}
        num_preds = len(scores)

        if num_preds > 0:
            instance_mask = np.zeros(
                (num_preds, *res["original_size_of_raw_data"]), dtype=np.uint8
            )
            for instance_id, (pbox, pscore, plabel) in enumerate(
                zip(boxes, scores, labels), start=0
            ):
                mask_slicing = [
                    slice(instance_id, instance_id + 1),
                    slice(int(pbox[0]) + 1, int(pbox[2])),
                    slice(int(pbox[1]) + 1, int(pbox[3])),
                ]
                if instance_mask.ndim == 4:
                    mask_slicing.append(slice(int(pbox[4]) + 1, int(pbox[5])))
                instance_mask[tuple(mask_slicing)] = 1

                prediction_meta[int(instance_id + 1)] = {
                    "score": float(pscore),
                    "label": int(plabel),
                    "box": list(map(int, pbox)),
                }
        else:
            instance_mask = np.zeros(
                (1, *res["original_size_of_raw_data"]), dtype=np.uint8
            )

        logger.info(f"Created instance mask with {num_preds} instances.")

        instance_mask = instance_mask.transpose(1, 2, 3, 0)
        instance_mask_itk = sitk.GetImageFromArray(instance_mask)
        instance_mask_itk.SetOrigin(res["itk_origin"])
        instance_mask_itk.SetDirection(res["itk_direction"])
        instance_mask_itk.SetSpacing(res["itk_spacing"])
        sitk.WriteImage(instance_mask_itk, str(save_dir / f"{cid}_boxes.nii.gz"))
        save_json(prediction_meta, save_dir / f"{cid}_boxes.json")


@env_guard
def masks2nii():
    """
    Only for visualisation purposes.
    """
    import argparse
    import os
    from pathlib import Path

    import numpy as np
    import SimpleITK as sitk
    from loguru import logger

    from nndet.io import load_pickle, save_json
    from nndet.io.paths import get_task, get_training_dir
    from nndet.utils.info import maybe_verbose_iterable

    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument("model", type=str, help="model name, e.g. RetinaUNetV0")
    parser.add_argument("fold", type=int, help="fold to sweep.")
    parser.add_argument(
        "-o",
        "--overwrites",
        type=str,
        nargs="+",
        help="overwrites for config file",
        required=False,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Minimum probability of predictions",
        required=False,
        default=0.5,
    )
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    model = args.model
    fold = args.fold
    task = args.task
    overwrites = args.overwrites
    test = args.test
    threshold = args.threshold

    task_name = get_task(task, name=True, models=True)
    task_dir = Path(os.getenv("det_models")) / task_name

    training_dir = get_training_dir(task_dir / model, fold)

    overwrites = overwrites if overwrites is not None else []
    overwrites.append("host.parent_data=${env:det_data}")
    overwrites.append("host.parent_results=${env:det_models}")

    prediction_dir = (
        training_dir / "test_predictions" if test else training_dir / "val_predictions"
    )
    save_dir = (
        training_dir / "test_predictions_nii"
        if test
        else training_dir / "val_predictions_nii"
    )
    save_dir.mkdir(exist_ok=True)

    case_ids = [p.stem.rsplit("_", 1)[0] for p in prediction_dir.glob("*_masks.npz")]
    case_ids.sort()
    for cid in maybe_verbose_iterable(case_ids):
        res = np.load(prediction_dir / f"{cid}_masks.npz")
        res_meta = load_pickle(prediction_dir / f"{cid}_masks.pkl")

        masks = res["pred_masks"]
        scores = res["pred_scores"]
        labels = res["pred_labels"]

        keep = scores >= threshold
        masks = masks[keep]
        scores = scores[keep]
        labels = labels[keep]

        idx = np.argsort(scores)
        masks = masks[idx]
        scores = scores[idx]
        labels = labels[idx]

        prediction_meta = {}
        for instance_id, (pscore, plabel) in enumerate(zip(scores, labels), start=1):
            prediction_meta[int(instance_id)] = {
                "score": float(pscore),
                "label": int(plabel),
            }

        logger.info(f"Created binary mask with {masks.shape[0]} instances.")

        if masks.size == 0:
            masks = np.zeros((1, *res_meta["original_size_of_raw_data"]))
        masks = masks.transpose(1, 2, 3, 0)
        instance_mask_itk = sitk.GetImageFromArray(masks)
        instance_mask_itk.SetOrigin(res_meta["itk_origin"])
        instance_mask_itk.SetDirection(res_meta["itk_direction"])
        instance_mask_itk.SetSpacing(res_meta["itk_spacing"])

        sitk.WriteImage(instance_mask_itk, str(save_dir / f"{cid}_masks.nii.gz"))
        save_json(prediction_meta, save_dir / f"{cid}_masks.json")


@env_guard
def seg2nii():
    """
    Only for visualisation purposes.
    """
    import argparse
    import os
    from pathlib import Path

    import SimpleITK as sitk

    from nndet.io import load_pickle
    from nndet.io.paths import get_task, get_training_dir
    from nndet.utils.info import maybe_verbose_iterable

    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument("model", type=str, help="model name, e.g. RetinaUNetV0")
    parser.add_argument("fold", type=int, help="experiment fold")
    parser.add_argument(
        "-o",
        "--overwrites",
        type=str,
        nargs="+",
        help="overwrites for config file",
        required=False,
    )
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    model = args.model
    fold = args.fold
    task = args.task
    overwrites = args.overwrites
    test = args.test

    task_name = get_task(task, name=True, models=True)
    task_dir = Path(os.getenv("det_models")) / task_name

    training_dir = get_training_dir(task_dir / model, fold)

    overwrites = overwrites if overwrites is not None else []
    overwrites.append("host.parent_data=${env:det_data}")
    overwrites.append("host.parent_results=${env:det_models}")

    prediction_dir = (
        training_dir / "test_predictions" if test else training_dir / "val_predictions"
    )
    save_dir = (
        training_dir / "test_predictions_nii"
        if test
        else training_dir / "val_predictions_nii"
    )
    save_dir.mkdir(exist_ok=True)

    case_ids = [p.stem.rsplit("_", 1)[0] for p in prediction_dir.glob("*_seg.pkl")]
    for cid in maybe_verbose_iterable(case_ids):
        res = load_pickle(prediction_dir / f"{cid}_seg.pkl")

        seg_itk = sitk.GetImageFromArray(res["pred_seg"])
        seg_itk.SetOrigin(res["itk_origin"])
        seg_itk.SetDirection(res["itk_direction"])
        seg_itk.SetSpacing(res["itk_spacing"])

        sitk.WriteImage(seg_itk, str(save_dir / f"{cid}_seg.nii.gz"))


def unpack():
    import argparse
    from pathlib import Path

    from nndet.io.load import unpack_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Path to folder to unpack")
    parser.add_argument(
        "num_processes", type=int, help="number of processes to use for unpacking"
    )
    args = parser.parse_args()
    p = args.path
    num_processes = args.num_processes
    unpack_dataset(p, num_processes, False)


def env():
    import os
    import sys

    import torch

    print("----- PyTorch Information -----")
    print(f"PyTorch Version: {torch.version.__version__}")
    print(f"PyTorch Debug: {torch.version.debug}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"PyTorch Backend cudnn: {torch.backends.cudnn.version()}")
    print(f"PyTorch CUDA Arch List: {torch.cuda.get_arch_list()}")
    print(f"PyTorch Current Device Capability: {torch.cuda.get_device_capability()}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print("\n")

    print("----- System Information -----")
    stream = os.popen("nvcc --version")
    output = stream.read()
    print(f"System NVCC: {output}")
    print(f"System Arch List: {os.getenv('TORCH_CUDA_ARCH_LIST', None)}")
    print(f"System OMP_NUM_THREADS: {os.getenv('OMP_NUM_THREADS', None)}")
    print(f"System CUDA_HOME is None: {os.getenv('CUDA_HOME', None) is None}")
    print(f"System CPU Count: {os.cpu_count()}")
    print(f"Python Version: {sys.version}")
    print("\n")

    print("----- nnDetection Information -----")
    print(f"det_num_threads {os.getenv('det_num_threads', None)}")
    print(f"det_data is set {os.getenv('det_data', None) is not None}")
    print(f"det_models is set {os.getenv('det_models', None) is not None}")
    print("\n")


def print_reg():
    """
    Helper function to print registry entries of nnDetection
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "registry", type=str, help="Name of registry, e.g. augmentation"
    )

    args = parser.parse_args()
    registry_name = args.registry
    registry_name = registry_name.lower()

    if registry_name == "augmentation":
        from nndet.io.augmentation import AUGMENTATION_REGISTRY

        registry = AUGMENTATION_REGISTRY
    elif registry_name == "dataloader":
        from nndet.io.datamodule import DATALOADER_REGISTRY

        registry = DATALOADER_REGISTRY
    elif registry_name == "planner":
        from nndet.planning import PLANNER_REGISTRY

        registry = PLANNER_REGISTRY
    elif registry_name == "module":
        from nndet.ptmodule import MODULE_REGISTRY

        registry = MODULE_REGISTRY
    elif registry_name == "optimizer":
        from nndet.ptmodule.optimizer import OPTIMIZER_REGISTRY

        registry = OPTIMIZER_REGISTRY
    else:
        raise ValueError(f"Did not find registry for {registry_name}")

    print(registry)


@env_guard
def create_test_split():
    import argparse
    import os
    import sys
    from pathlib import Path

    from loguru import logger

    from nndet.io.prepare import create_test_split
    from nndet.utils.config import load_dataset_info

    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument("--size", type=float, help="Size of test split", default=0.3)

    args = parser.parse_args()
    task = args.task
    test_size = args.size

    task_name = get_task(task, name=True)
    task_dir = Path(os.getenv("det_data")) / task_name
    raw_splitted_dir = task_dir / "raw_splitted"

    logger.remove()
    logger.add(sys.stdout, format="{level} {message}", level="DEBUG")
    logger.add(raw_splitted_dir.parent / "split.log", level="DEBUG")

    meta = load_dataset_info(task_dir)

    create_test_split(
        raw_splitted_dir,
        num_modalities=len(meta["modalities"]),
        test_size=test_size,
        random_state=0,
        shuffle=True,
    )


@env_guard
def create_split():
    """
    Utility function to create (best effort splits)

    This function will automatically generate splits which can be used
    inside nndetection. In order to properly do this, the case names need to
    follow this convention:

        {patient id}_{session id}_{modality id}.{data extension}

        Alterantivly, if `with_patients` is not used the
        {patient id}_{modality id}.{data extension}

        notation can be used as well

    - patient id: this represents a patient identifier, e.g. one patient who
        was scanned two times will have the same patient id
    - session id: the session id is an identifier to differentiate multiple
        scans of the same patient
    - modality id [only for data images]: identify the modality (or sequence)
        of the data channel
    - data extension: refers to the data type: .nii.gz for data and
        segmentation files, json for label files

    The case id = patient id + session id need to unique across the whole
    dataset! The patient names are not allowed to contain any other underscores
    "_"!

    (if not manually disabled) the splits will automatically ensure that the
    same patient is only present in a single fold. Furthermore, it will
    try to stratify the classes between folds (priority will be given to
    rare classes -> this is necessary e.g. when patients can contain more
    than one class)
    """
    import argparse
    import os
    import sys
    from pathlib import Path

    import numpy as np
    from loguru import logger
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    from nndet.io import load_json, save_json, save_pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds")
    parser.add_argument(
        "--with_patients",
        action="store_true",
        help="Derive patient information from names.",
    )

    args = parser.parse_args()
    task = args.task
    num_folds = args.num_folds
    with_patients = args.with_patients

    task_name = get_task(task, name=True)
    task_dir = Path(os.getenv("det_data")) / task_name

    if not task_dir.is_dir():
        raise ValueError(f"{task_dir} is not a valid task directory!")

    preprocessed_dir = task_dir / "preprocessed"
    preprocessed_dir.mkdir(exist_ok=True)

    raw_splitted_dir = task_dir / "raw_splitted"
    if not raw_splitted_dir.is_dir():
        raise ValueError(f"{raw_splitted_dir} is not a directory!")
    label_dir = raw_splitted_dir / "labelsTr"
    if not label_dir.is_dir():
        raise ValueError(f"{label_dir} is not a directory!")

    splits_path_json = preprocessed_dir / "splits_final.json"
    splits_path_pkl = preprocessed_dir / "splits_final.pkl"

    if splits_path_json.is_file():
        raise ValueError(f"{splits_path_json} already exists.")
    if splits_path_pkl.is_file():
        raise ValueError(f"{splits_path_pkl} already exists.")

    # setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(task_dir / "split.log", level="DEBUG")

    # parse case ids
    case_ids = [p.stem for p in label_dir.glob("*") if p.suffix == ".json"]
    case_ids = sorted(case_ids)

    # split into pid and sid
    for cid in case_ids:
        if len(cid.split("_")) > 2:
            raise ValueError(
                f"{cid} does not follow the naming convention please read the docs."
            )

    if with_patients:
        patient_ids = [cid.split("_")[0] for cid in case_ids]
        # session_ids = [cid.split("_")[1] for cid in case_ids]
        logger.info(
            f"Parsed {len(case_ids)} case ids and {len(set(patient_ids))} unique patient ids \n{case_ids}"
        )
    else:
        patient_ids = None
        logger.info(f"Parsed {len(case_ids)} case ids \n{case_ids}")

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
    if patient_ids is not None:
        cv = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=0)
    else:
        cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=0)

    for fold_idx, (train_idx, val_idx) in enumerate(
        cv.split(case_ids, reduced_classes, patient_ids)
    ):
        train_cids = [case_ids[_i] for _i in train_idx]
        val_cids = [case_ids[_i] for _i in val_idx]
        intersection_cids = set(train_cids).intersection(val_cids)

        assert not intersection_cids
        logger.info(
            f"Generated fold {fold_idx} with {len(train_cids)} "
            f"train {len(val_cids)} val cases. "
            f"Intersection {intersection_cids} (should be empty)"
        )

        splits.append({"train": train_cids, "val": val_cids})

    # save splits
    save_json(splits, splits_path_json)
    save_pickle(splits, splits_path_pkl)


if __name__ == "__main__":
    # env()
    masks2nii()
