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

import os
import random
import argparse
from pathlib import Path
from multiprocessing import Pool
from itertools import repeat

import numpy as np
import SimpleITK as sitk
from loguru import logger

from nndet.io import save_json
from nndet.utils.check import env_guard


# # 2D example
# [Ignore, Not supported]
# dim = 2
# image_size = [512, 512]
# object_size = [32, 64]
# object_width = 6
# num_images_tr = 100
# num_images_ts = 100

# 3D example

dim = 3
image_size = [256, 256, 256]
object_size = [16, 32]
object_width = 4


def generate_image(image_dir, label_dir, idx):
    random.seed(idx)
    np.random.seed(idx)

    logger.info(f"Generating case_{idx}")
    selected_size = np.random.randint(object_size[0], object_size[1])
    selected_class = np.random.randint(0, 2)

    data = np.random.rand(*image_size)
    mask = np.zeros_like(data)

    top_left = [np.random.randint(0, image_size[i] - selected_size) for i in range(dim)]

    if selected_class == 0:
        slicing = tuple([slice(tp, tp + selected_size) for tp in top_left])
        data[slicing] = data[slicing] + 0.4
        data = data.clip(0, 1)
        mask[slicing] = 1
    elif selected_class == 1:
        slicing = tuple([slice(tp, tp + selected_size) for tp in top_left])

        inner_slicing = [slice(tp + object_width, tp + selected_size - object_width) for tp in top_left]
        if len(inner_slicing) == 3:
            inner_slicing[0] = slice(0, image_size[0])
        inner_slicing = tuple(inner_slicing)

        object_mask = np.zeros_like(mask).astype(bool)
        object_mask[slicing] = 1
        object_mask[inner_slicing] = 0

        data[object_mask] = data[object_mask] + 0.4
        data = data.clip(0, 1)
        mask[object_mask] = 1
    else:
        raise NotImplementedError

    if dim == 2:
        data = data[None]
        mask = mask[None]

    data_itk = sitk.GetImageFromArray(data)
    mask_itk = sitk.GetImageFromArray(mask)
    mask_meta = {
        "instances": {
            "1": selected_class
        },
    }
    sitk.WriteImage(data_itk, str(image_dir / f"case_{idx}_0000.nii.gz"))
    sitk.WriteImage(mask_itk, str(label_dir / f"case_{idx}.nii.gz"))
    save_json(mask_meta, label_dir / f"case_{idx}.json")


@env_guard
def main():
    """
    Generate an example dataset for nnDetection to test the installation or
    experiment with ideas.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--full',
        help="Increase size of dataset. "
        "Default sizes train/test 10/10 and full 1000/1000.",
        action='store_true',
        )
    parser.add_argument(
        '--num_processes',
        help="Use multiprocessing to create dataset.",
        type=int,
        default=0,
        )
    args = parser.parse_args()

    full = args.full
    num_processes = args.num_processes

    num_images_tr = 1000 if full else 10
    num_images_ts = 1000 if full else 10

    meta = {
        "task": f"Task000D{dim}_Example",
        "name": "Example",
        "target_class": None,
        "test_labels": True,
        "labels": {"0": "Square", "1": "SquareHole"},
        "modalities": {"0": "MRI"},
        "dim": dim,
    }

    # setup paths
    data_task_dir = Path(os.getenv("det_data")) / meta["task"]
    data_task_dir.mkdir(parents=True, exist_ok=True)
    save_json(meta, data_task_dir / "dataset.json")

    raw_splitted_dir = data_task_dir / "raw_splitted"
    images_tr_dir = raw_splitted_dir / "imagesTr"
    images_tr_dir.mkdir(parents=True, exist_ok=True)
    labels_tr_dir = raw_splitted_dir / "labelsTr"
    labels_tr_dir.mkdir(parents=True, exist_ok=True)
    images_ts_dir = raw_splitted_dir / "imagesTs"
    images_ts_dir.mkdir(parents=True, exist_ok=True)
    labels_ts_dir = raw_splitted_dir / "labelsTs"
    labels_ts_dir.mkdir(parents=True, exist_ok=True)

    if num_processes == 0:
        for idx in range(num_images_tr):
            generate_image(
                images_tr_dir,
                labels_tr_dir,
                idx,
                )

        for idx in range(num_images_tr, num_images_tr + num_images_ts):
            generate_image(
                images_ts_dir,
                labels_ts_dir,
                idx,
                )
    else:
        logger.info("Using multiprocessing to create example dataset.")
        with Pool(processes=num_processes) as p:
            p.starmap(
                generate_image,
                zip(
                    repeat(images_tr_dir),
                    repeat(labels_tr_dir),
                    range(num_images_tr),
                )
            )
        with Pool(processes=num_processes) as p:
            p.starmap(
                generate_image,
                zip(
                    repeat(images_ts_dir),
                    repeat(labels_ts_dir),
                    range(num_images_tr, num_images_tr + num_images_ts),
                )
            )


if __name__ == '__main__':
    main()
