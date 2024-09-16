# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import List

import numpy as np

from nndet.io.paths import get_task
from nndet.utils.check import env_guard


@env_guard
def boxes2mitk():
    """
    Only for visualisation purposes.
    """
    import argparse
    import os
    from pathlib import Path

    import numpy as np
    from loguru import logger

    from nndet.io import load_pickle, save_json
    from nndet.io.paths import get_task, get_training_dir
    from nndet.utils.info import maybe_verbose_iterable

    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument("model", type=str, help="model name, e.g. RetinaUNetV0")
    parser.add_argument("fold", type=int, help="experiment fold")
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
    test = args.test
    threshold = args.threshold

    task_name = get_task(task, name=True, models=True)
    task_dir = Path(os.getenv("det_models")) / task_name

    training_dir = get_training_dir(task_dir / model, fold)

    prediction_dir = training_dir / "test_predictions" if test else training_dir / "val_predictions"
    save_dir = training_dir / "test_predictions_nii" if test else training_dir / "val_predictions_nii"
    save_dir.mkdir(exist_ok=True)

    case_ids = [p.stem.rsplit("_", 1)[0] for p in prediction_dir.glob("*_boxes.pkl")]
    case_ids.sort()
    for cid in maybe_verbose_iterable(case_ids):
        res = load_pickle(prediction_dir / f"{cid}_boxes.pkl")
        boxes = res["pred_boxes"]
        scores = res["pred_scores"]
        labels = res["pred_labels"]

        # res["itk_direction"]
        mitk_json = {
            "FileFormat": "MITK ROI",
            "Version": 1,
            "Caption": "{label}: {score}",
            "Geometry": {
                "Origin": res["itk_origin"],
                "Spacing": res["itk_spacing"],
                "Size": res["original_size_of_raw_data"].tolist()[::-1],
            },
            "ROIs": [],
        }

        # filter predictions
        _mask = scores >= threshold
        boxes = boxes[_mask]
        labels = labels[_mask]
        scores = scores[_mask]

        idx = np.argsort(scores)
        scores = scores[idx]
        boxes = boxes[idx]
        labels = labels[idx]

        _dtype = float
        for instance_id, (pbox, pscore, plabel) in enumerate(zip(boxes, scores, labels), start=1):
            mitk_json["ROIs"].append(
                {
                    "ID": instance_id,
                    "Min": [_dtype(pbox[0]), _dtype(pbox[1]), _dtype(pbox[4])][::-1],
                    "Max": [_dtype(pbox[2]), _dtype(pbox[3]), _dtype(pbox[5])][::-1],
                    "Properties": {
                        "ColorProperty": {"color": [1, 0, 0]},  # color of bounding box
                        "FloatProperty": {
                            "score": round(float(pscore), 2),
                            "label": float(plabel),
                            "lineWidth": 2,  # line width of bounding box
                        },
                    },
                }
            )
        logger.info(f"Created prediction {cid} with {len(mitk_json['ROIs'])} instances.")
        save_json(mitk_json, save_dir / f"{cid}_boxes_mitk.json")


@env_guard
def boxes2mitkv2():
    """
    Only for visualisation purposes.
    MITKv2 Format
    """
    import argparse
    import os
    from pathlib import Path

    import numpy as np
    from loguru import logger

    from nndet.io import load_pickle, save_json
    from nndet.io.paths import get_task, get_training_dir
    from nndet.utils.info import maybe_verbose_iterable

    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument("model", type=str, help="model name, e.g. RetinaUNetV0")
    parser.add_argument("fold", type=int, help="experiment fold")
    parser.add_argument(
        "--threshold",
        type=float,
        help="Minimum probability of predictions",
        required=False,
        default=0.5,
    )
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--publication",
        action="store_true",
        help="Produces an extra set of files with red boxes without labels e.g. for publication",
    )

    args = parser.parse_args()
    model = args.model
    fold = args.fold
    task = args.task
    test = args.test
    threshold = args.threshold
    publication = args.publication

    task_name = get_task(task, name=True, models=True)
    task_dir = Path(os.getenv("det_models")) / task_name

    training_dir = get_training_dir(task_dir / model, fold)

    prediction_dir = training_dir / "test_predictions" if test else training_dir / "val_predictions"
    save_dir = training_dir / "test_predictions_nii" if test else training_dir / "val_predictions_nii"
    save_dir.mkdir(exist_ok=True)

    case_ids = [p.stem.rsplit("_", 1)[0] for p in prediction_dir.glob("*_boxes.pkl")]
    case_ids.sort()
    for cid in maybe_verbose_iterable(case_ids):
        res = load_pickle(prediction_dir / f"{cid}_boxes.pkl")
        boxes = res["pred_boxes"]
        scores = res["pred_scores"]
        labels = res["pred_labels"]

        img_size = res["original_size_of_raw_data"].tolist()[::-1]
        origin = np.array(res["itk_origin"])
        spacing = np.array(res["itk_spacing"])
        direction = np.array(res["itk_direction"]).reshape((3, 3))
        spacing_matrix = np.diag(spacing)
        rotation_scaling = np.dot(direction, spacing_matrix)

        # Create the full transformation matrix (4x4)
        transform = np.eye(4)
        transform[:3, :3] = rotation_scaling
        transform[3, :3] = origin
        transform = transform.reshape(-1).tolist()

        mitk_json = {
            "FileFormat": "MITK ROI",
            "Version": 2,
            "Caption": "{label}|{score}",
            "Geometry": {
                "Size": img_size,
                "Transform": transform,
            },
            "ROIs": [],
        }

        # filter predictions
        _mask = scores >= threshold
        boxes = boxes[_mask]
        labels = labels[_mask]
        scores = scores[_mask]

        idx = np.argsort(scores)
        scores = scores[idx]
        boxes = boxes[idx]
        labels = labels[idx]

        _dtype = float
        rois = []
        for instance_id, (pbox, pscore, plabel) in enumerate(zip(boxes, scores, labels), start=1):
            rois.append(
                {
                    "ID": instance_id,
                    "Min": [_dtype(pbox[0]), _dtype(pbox[1]), _dtype(pbox[4])][::-1],
                    "Max": [_dtype(pbox[2]), _dtype(pbox[3]), _dtype(pbox[5])][::-1],
                    "Properties": {
                        "ColorProperty": {"color": [1, 1, 1]},  # color of bounding box
                        "FloatProperty": {
                            "score": round(float(pscore), 2),
                            "label": float(plabel),
                            "lineWidth": 2,  # line width of bounding box
                        },
                    },
                }
            )
        mitk_json["ROIs"] = rois

        logger.info(f"Created prediction {cid} with {len(mitk_json['ROIs'])} instances.")
        save_json(mitk_json, save_dir / f"{cid}_boxes_mitkv2.json")

        if publication:
            mitk_json["Caption"] = ""  # remove caption
            # set all boxes to red
            for roi_idx in range(len(rois)):
                rois[roi_idx]["Properties"]["ColorProperty"]["color"] = [1, 0, 0]
            mitk_json["ROIs"] = rois
            save_json(mitk_json, save_dir / f"{cid}_boxes_mitkv2_publication.json")


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
    test = args.test
    threshold = args.threshold

    task_name = get_task(task, name=True, models=True)
    task_dir = Path(os.getenv("det_models")) / task_name

    training_dir = get_training_dir(task_dir / model, fold)

    prediction_dir = training_dir / "test_predictions" if test else training_dir / "val_predictions"
    save_dir = training_dir / "test_predictions_nii" if test else training_dir / "val_predictions_nii"
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
        for instance_id, (pbox, pscore, plabel) in enumerate(zip(boxes, scores, labels), start=1):
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

        logger.info(f"Created instance mask {cid} with {instance_mask.max()} instances.")

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
    test = args.test
    threshold = args.threshold

    task_name = get_task(task, name=True, models=True)
    task_dir = Path(os.getenv("det_models")) / task_name

    training_dir = get_training_dir(task_dir / model, fold)

    prediction_dir = training_dir / "test_predictions" if test else training_dir / "val_predictions"
    save_dir = training_dir / "test_predictions_nii2" if test else training_dir / "val_predictions_nii2"
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
            instance_mask = np.zeros((num_preds, *res["original_size_of_raw_data"]), dtype=np.uint8)
            for instance_id, (pbox, pscore, plabel) in enumerate(zip(boxes, scores, labels), start=0):
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
            instance_mask = np.zeros((1, *res["original_size_of_raw_data"]), dtype=np.uint8)

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
    test = args.test
    threshold = args.threshold

    task_name = get_task(task, name=True, models=True)
    task_dir = Path(os.getenv("det_models")) / task_name

    training_dir = get_training_dir(task_dir / model, fold)

    prediction_dir = training_dir / "test_predictions" if test else training_dir / "val_predictions"
    save_dir = training_dir / "test_predictions_nii" if test else training_dir / "val_predictions_nii"
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
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    model = args.model
    fold = args.fold
    task = args.task
    test = args.test

    task_name = get_task(task, name=True, models=True)
    task_dir = Path(os.getenv("det_models")) / task_name

    training_dir = get_training_dir(task_dir / model, fold)

    prediction_dir = training_dir / "test_predictions" if test else training_dir / "val_predictions"
    save_dir = training_dir / "test_predictions_nii" if test else training_dir / "val_predictions_nii"
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
    parser.add_argument("num_processes", type=int, help="number of processes to use for unpacking")
    parser.add_argument("--data_float16", action="store_true", help="Convert data to float16")
    parser.add_argument("--label_int8", action="store_true", help="Convert label to int8")

    args = parser.parse_args()
    p = args.path
    num_processes = args.num_processes
    data_float16: bool = args.data_float16
    label_int8: bool = args.label_int8

    data_dtype = np.float16 if data_float16 else None
    label_dtype = np.int8 if label_int8 else None

    if data_float16 or label_int8:
        print("WARNING: Use at your own risk. Manual dtypes set, no additional check will be performed.")

    unpack_dataset(
        p,
        processes=num_processes,
        delete_npz=False,
        data_dtype=data_dtype,
        seg_dtype=label_dtype,
    )


@env_guard
def unpack_task():
    import argparse

    from nndet.io.load import unpack_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument(
        "data_identifiers",
        type=str,
        nargs="+",
        help="Data identifiers to unpack, e.g. D3V001_3d and D3V001_3dlr1",
    )
    parser.add_argument(
        "-p",
        "--num_processes",
        type=int,
        help="Number of processes to use for unpacking",
        default=6,
        required=False,
    )
    parser.add_argument("--data_float16", action="store_true", help="Convert data to float16")
    parser.add_argument("--label_int8", action="store_true", help="Convert label to int8")
    args = parser.parse_args()

    task: str = args.task
    data_identifiers: List[str] = args.data_identifiers
    num_processes: int = args.num_processes
    data_float16: bool = args.data_float16
    label_int8: bool = args.label_int8

    data_dtype = np.float16 if data_float16 else None
    label_dtype = np.int8 if label_int8 else None

    task_path = get_task(task)
    preprocessed_path = task_path / "preprocessed"
    if not preprocessed_path.is_dir():
        raise ValueError(f"Expected {preprocessed_path} to exist, please run preprocessing first.")

    for di in data_identifiers:
        _data_identifier_path = preprocessed_path / di
        if not _data_identifier_path.is_dir():
            raise ValueError(f"{di} is not a valid data identifier since {_data_identifier_path} does not exist")

        if data_float16 or label_int8:
            print("WARNING: Use at your own risk. Manual dtypes set, no additional check will be performed.")

        unpack_dataset(
            _data_identifier_path / "imagesTr",
            processes=num_processes,
            delete_npz=False,
            data_dtype=data_dtype,
            seg_dtype=label_dtype,
        )


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
    parser.add_argument("registry", type=str, help="Name of registry, e.g. augmentation")

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
def create_test_data_split():
    """
    Random test data split -> no stratification
    """
    import argparse
    import os
    import sys
    from datetime import datetime
    from pathlib import Path

    from loguru import logger

    from nndet.io.prepare import create_test_split
    from nndet.utils.config import load_dataset_info

    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument("--size", type=float, help="Size of test split", default=0.3)
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Enable a best effort stratification of patients.",
    )

    args = parser.parse_args()
    task = args.task
    test_size = args.size
    stratify = args.stratify

    task_name = get_task(task, name=True)
    task_dir = Path(os.getenv("det_data")) / task_name
    raw_splitted_dir = task_dir / "raw_splitted"

    logger.remove()
    logger.add(sys.stdout, format="{level} {message}", level="DEBUG")
    logger.add(raw_splitted_dir.parent / "split.log", level="DEBUG")

    current_time = datetime.now()
    current_time_str = current_time.strftime("%d/%m/%Y %H:%M:%S")
    logger.info(f"+++ Running nndet_test_split {current_time_str} +++")

    meta = load_dataset_info(task_dir)
    session_id = meta.get("session_id", False)
    if session_id:
        _error_str = "Session id is enabled, which is supported in this script. Please create the test set manually!"
        logger.error(_error_str)
        raise RuntimeError(_error_str)

    create_test_split(
        raw_splitted_dir,
        num_modalities=len(meta["modalities"]),
        test_size=test_size,
        random_state=0,
        shuffle=True,
        do_stratify=stratify,
    )


@env_guard
def create_cv_split():
    """
    Utility function to create (best effort) cross validation splits

    This function will automatically generate splits which can be used
    inside nndetection. In order to properly do this, the case names need to
    follow this convention:

        if `with_patients` is used:
        {patient id}_{session id}_{modality id}.{data extension}

        otherwise:
        {patient id}_{modality id}.{data extension}

    - patient id: this represents a patient identifier, e.g. one patient who
        was scanned two times will have the same patient id
    - session id: the session id is an identifier to differentiate multiple
        scans of the same patient
    - modality id [only for data images]: identify the modality (or sequence)
        of the data channel
    - data extension: refers to the data type: .nii.gz for data and
        segmentation files, json for label files

    The case id (which is equal to patient id + session id) needs to be unique
    across the entire dataset! The patient names are not allowed to contain
    any other underscores "_"!

    If `with_patients` is used, the splits will automatically ensure that the
    same patient is only present in a single fold. Furthermore, it will
    try to stratify the classes between folds (priority will be given to
    rare classes -> this is necessary e.g. when patients can contain more
    than one class)
    """
    import argparse
    import os
    import sys
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    from loguru import logger
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    from nndet.io import load_json, save_json, save_pickle
    from nndet.utils.config import load_dataset_info

    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds")

    args = parser.parse_args()
    task = args.task
    num_folds = args.num_folds

    task_name = get_task(task, name=True)
    task_dir = Path(os.getenv("det_data")) / task_name
    dataset_info = load_dataset_info(task_dir)
    with_patients = dataset_info.get("session_id", False)
    session_id_str = "enabled" if with_patients else "disabled"
    logger.info(f"Running cv splits with session_id: {session_id_str}")

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
    logger.add(
        sys.stdout,
        format="<level>{level}</level>: {message}",
        level="INFO",
        colorize=True,
    )
    logger.add(task_dir / "split.log", level="DEBUG")

    current_time = datetime.now()
    current_time_str = current_time.strftime("%d/%m/%Y %H:%M:%S")
    logger.info(f"+++ Running nndet_cv_split {current_time_str} +++")

    # parse case ids
    case_ids = [p.stem for p in label_dir.glob("*") if p.suffix == ".json"]
    case_ids = sorted(case_ids)

    # split into pid and sid
    for cid in case_ids:
        if len(cid.split("_")) > 2:
            raise ValueError(f"{cid} does not follow the naming convention please read the docs.")

    if with_patients:
        patient_ids = [cid.split("_")[0] for cid in case_ids]
        # session_ids = [cid.split("_")[1] for cid in case_ids]
        logger.info(f"Parsed {len(case_ids)} case ids and {len(set(patient_ids))} unique patient ids \n{case_ids}")
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

    # save splits
    save_json(splits, splits_path_json)
    save_pickle(splits, splits_path_pkl)


@env_guard
def splits_pkl_to_json():
    import argparse
    import os
    from pathlib import Path

    from nndet.io import load_pickle, save_json

    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument(
        "--splits_name",
        type=str,
        help="Name of splits file",
        required=False,
        default="splits_final",
    )

    args = parser.parse_args()
    task = args.task
    splits_name = args.splits_name

    task_name = get_task(task, name=True)
    task_dir = Path(os.getenv("det_data")) / task_name

    if not task_dir.is_dir():
        raise ValueError(f"{task_dir} is not a valid task directory!")
    preprocessed_dir = task_dir / "preprocessed"
    if not preprocessed_dir.is_dir():
        raise ValueError(f"{preprocessed_dir} is not a directory!")

    splits_path_json = preprocessed_dir / f"{splits_name}.json"
    splits_path_pkl = preprocessed_dir / f"{splits_name}.pkl"

    if not splits_path_pkl.is_file():
        raise ValueError(f"{splits_path_pkl} is not a valid splits file!")

    print(f"Converting {splits_path_pkl} to {splits_path_json}")
    splits = load_pickle(splits_path_pkl)

    splits_no_array = []
    for fold in splits:
        splits_no_array.append({k: list(x) if isinstance(x, np.ndarray) else x for k, x in fold.items()})
    save_json(splits_no_array, splits_path_json)


@env_guard
def numpy2blosc():
    import argparse
    import os
    import shutil
    from pathlib import Path

    from loguru import logger

    from nndet.io import load_json, load_pickle, save_json, save_pickle
    from nndet.io.dataformat import data_format_to_class_mapping
    from nndet.utils.info import maybe_verbose_iterable

    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument("metafile", type=str, help="D3V002_3d")
    parser.add_argument("--process_npy", action="store_true")
    parser.add_argument("--process_npz", action="store_true")

    args = parser.parse_args()
    task = args.task
    metafile = args.metafile
    process_npy = args.process_npy
    process_npz = args.process_npz if args.process_npy else (not process_npy)

    task_name = get_task(task, name=True)
    task_dir = Path(os.getenv("det_data")) / task_name

    if not task_dir.is_dir():
        raise ValueError(f"{task_dir} is not a valid task directory!")
    preprocessed_dir = task_dir / "preprocessed"
    if not preprocessed_dir.is_dir():
        raise ValueError(f"{preprocessed_dir} is not a directory!")
    if not (preprocessed_dir / f"{metafile}.json").is_file():
        raise ValueError(f"{metafile}.json not found!")

    meta_json = load_json(preprocessed_dir / f"{metafile}.json")
    meta_pkl = load_pickle(preprocessed_dir / f"{metafile}.pkl")

    data_identifier = meta_json["data_identifier"]
    patch_size = meta_json["patch_size"]

    npx_dir = preprocessed_dir / data_identifier
    blosc_dir = preprocessed_dir / "D3V001Blosc_3d"

    npx_handler = data_format_to_class_mapping["npz"]
    blosc_handler = data_format_to_class_mapping["b2nd"]

    for mode in ["Tr", "Ts"]:
        npx_imgdir = npx_dir / f"images{mode}"
        blosc_imgdir = blosc_dir / f"images{mode}"

        npx_lbldir = npx_dir / f"labels{mode}"
        blosc_lbldir = blosc_dir / f"labels{mode}"

        if npx_imgdir.is_dir():
            logger.info(f"Converting {'npz' if process_npz else 'npy'} files to blosc")
            blosc_imgdir.mkdir(parents=True, exist_ok=True)
            if process_npz:
                cases = [p.stem for p in npx_imgdir.glob("*.npz")]
                for case in maybe_verbose_iterable(cases):
                    data = npx_handler.load_data(npx_imgdir / f"{case}.npz")
                    seg = npx_handler.load_seg(npx_imgdir / f"{case}.npz")
                    blosc_handler.save(blosc_imgdir / case, data, seg, patch_size=patch_size)
            else:
                cases = [p.stem for p in npx_imgdir.glob("*.npy") if "_seg" not in p.stem]
                for case in maybe_verbose_iterable(cases):
                    data = npx_handler.load_data(npx_imgdir / f"{case}.npy")
                    seg = npx_handler.load_seg(npx_imgdir / f"{case}_seg.npy")
                    blosc_handler.save(blosc_imgdir / case, data, seg, patch_size=patch_size)

            pkls = [p.name for p in npx_imgdir.glob("*.pkl")]
            for pkl in pkls:
                shutil.copy(npx_imgdir / pkl, blosc_imgdir / pkl)
        else:
            logger.warning(f"No images{mode} folder found!")

        if npx_lbldir.is_dir():
            shutil.copytree(npx_lbldir, blosc_lbldir, dirs_exist_ok=True)
        else:
            logger.warning(f"No labels{mode} folder found!")

    meta_json["planner_id"] = "D3V002Blosc"
    meta_json["preprocessed_data_format"] = "b2nd"
    meta_json["data_identifier"] = "D3V001Blosc_3d"
    save_json(meta_json, preprocessed_dir / "D3V002Blosc_3d.json")

    meta_pkl["planner_id"] = "D3V002Blosc"
    meta_pkl["preprocessed_data_format"] = "b2nd"
    meta_pkl["data_identifier"] = "D3V001Blosc_3d"
    save_pickle(meta_pkl, preprocessed_dir / "D3V002Blosc_3d.pkl")


if __name__ == "__main__":
    env()
