"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

def boxes2nii():
    import os
    import argparse
    from pathlib import Path

    import numpy as np
    import SimpleITK as sitk
    from loguru import logger

    from nndet.io import save_json, load_pickle
    from nndet.io.paths import get_task, get_training_dir
    from nndet.utils.info import maybe_verbose_iterable

    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument('model', type=str, help="model name, e.g. RetinaUNetV0")
    parser.add_argument('-f', '--fold', type=int, help="fold to sweep.", default=0, required=False)
    parser.add_argument('-o', '--overwrites', type=str, nargs='+',
                        help="overwrites for config file",
                        required=False)
    parser.add_argument('--threshold',
                        type=float,
                        help="Minimum probability of predictions",
                        required=False,
                        default=0.5,
                        )
    parser.add_argument('--test', action='store_true')

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

    prediction_dir = training_dir / "test_predictions" \
        if test else training_dir / "val_predictions"
    save_dir = training_dir / "test_predictions_nii" \
        if test else training_dir / "val_predictions_nii"
    save_dir.mkdir(exist_ok=True)

    case_ids = [p.stem.rsplit('_', 1)[0] for p in prediction_dir.glob("*_boxes.pkl")]
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
            mask_slicing = [slice(int(pbox[0]), int(pbox[2])),
                            slice(int(pbox[1]), int(pbox[3])),
                            ]
            if instance_mask.ndim == 3:
                mask_slicing.append(slice(int(pbox[4]), int(pbox[5])))
            instance_mask[tuple(mask_slicing)] = instance_id

            prediction_meta[int(instance_id)] = {
                "score": float(pscore),
                "label": int(plabel),
                "box": list(map(int, pbox))
            }

        logger.info(f"Created instance mask with {instance_mask.max()} instances.")

        instance_mask_itk = sitk.GetImageFromArray(instance_mask)
        instance_mask_itk.SetOrigin(res["itk_origin"])
        instance_mask_itk.SetDirection(res["itk_direction"])
        instance_mask_itk.SetSpacing(res["itk_spacing"])

        sitk.WriteImage(instance_mask_itk, str(save_dir / f"{cid}_boxes.nii.gz"))
        save_json(prediction_meta, save_dir / f"{cid}_boxes.json")


def seg2nii():
    import os
    import argparse
    from pathlib import Path

    import SimpleITK as sitk

    from nndet.io import load_pickle
    from nndet.io.paths import get_task, get_training_dir
    from nndet.utils.info import maybe_verbose_iterable

    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help="Task id e.g. Task12_LIDC OR 12 OR LIDC")
    parser.add_argument('model', type=str, help="model name, e.g. RetinaUNetV0")
    parser.add_argument('-f', '--fold', type=int, help="fold to sweep.", default=0, required=False)
    parser.add_argument('-o', '--overwrites', type=str, nargs='+',
                        help="overwrites for config file",
                        required=False)
    parser.add_argument('--test', action='store_true')

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

    prediction_dir = training_dir / "test_predictions" \
        if test else training_dir / "val_predictions"
    save_dir = training_dir / "test_predictions_nii" \
        if test else training_dir / "val_predictions_nii"
    save_dir.mkdir(exist_ok=True)

    case_ids = [p.stem.rsplit('_', 1)[0] for p in prediction_dir.glob("*_seg.pkl")]
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
    parser.add_argument('path', type=Path, help="Path to folder to unpack")
    parser.add_argument('num_processes', type=int, help="number of processes to use for unpacking")
    args = parser.parse_args()
    p = args.path
    num_processes = args.num_processes
    unpack_dataset(p, num_processes, False)
