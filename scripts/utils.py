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
    """
    Only for visualisation purposes.
    """
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
    overwrites.append("host.parent_data=${oc.env:det_data}")
    overwrites.append("host.parent_results=${oc.env:det_models}")

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
            mask_slicing = [slice(int(pbox[0]) + 1, int(pbox[2])),
                            slice(int(pbox[1]) + 1, int(pbox[3])),
                            ]
            if instance_mask.ndim == 3:
                mask_slicing.append(slice(int(pbox[4]) + 1, int(pbox[5])))
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
    """
    Only for visualisation purposes.
    """
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
    overwrites.append("host.parent_data=${oc.env:det_data}")
    overwrites.append("host.parent_results=${oc.env:det_models}")

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


def hydra_searchpath():
    from hydra import compose as hydra_compose
    from hydra import initialize_config_module
        
    initialize_config_module(config_module="nndet.conf")
    cfg = hydra_compose("config.yaml", return_hydra_config=True)

    print("Found config sources::")
    print("----------------------")
    for s in cfg.hydra.runtime.config_sources:
        print(s)


def env():
    import os
    import torch
    import sys
    print(f"----- PyTorch Information -----")
    print(f"PyTorch Version: {torch.version.__version__}")
    print(f"PyTorch Debug: {torch.version.debug}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"PyTorch Backend cudnn: {torch.backends.cudnn.version()}")
    print(f"PyTorch CUDA Arch List: {torch.cuda.get_arch_list()}")
    print(f"PyTorch Current Device Capability: {torch.cuda.get_device_capability()}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print("\n")

    print(f"----- System Information -----")
    stream = os.popen('nvcc --version')
    output = stream.read()
    print(f"System NVCC: {output}")
    print(f"System Arch List: {os.getenv('TORCH_CUDA_ARCH_LIST', None)}")
    print(f"System OMP_NUM_THREADS: {os.getenv('OMP_NUM_THREADS', None)}")
    print(f"System CUDA_HOME is None: {os.getenv('CUDA_HOME', None) is None}")
    print(f"System CPU Count: {os.cpu_count()}")
    print(f"Python Version: {sys.version}")
    print("\n")

    print(f"----- nnDetection Information -----")
    print(f"det_num_threads {os.getenv('det_num_threads', None)}")
    print(f"det_data is set {os.getenv('det_data', None) is not None}")
    print(f"det_models is set {os.getenv('det_models', None) is not None}")
    print("\n")


if __name__ == '__main__':
    env()
