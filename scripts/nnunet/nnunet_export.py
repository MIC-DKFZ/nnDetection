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

import copy
import os
import argparse
import sys

from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf
from hydra import initialize_config_module

from nnunet.paths import nnUNet_raw_data

from nndet.io import get_task
from nndet.utils.config import compose
from nndet.utils.nnunet import Exporter


def run(cfg, target_dir, stuff: bool):
    base_dir = Path(cfg.host.splitted_4d_output_dir)
    target_dir.mkdir(exist_ok=True, parents=True)

    if (base_dir / "imagesTs").is_dir():
        logger.info("Found test images and will export them too")
        ts_image_dir = base_dir / "imagesTs"
    else:
        ts_image_dir = None

    exporter = Exporter(data_info=OmegaConf.to_container(cfg.data),
                        tr_image_dir=base_dir / "imagesTr",
                        ts_image_dir=ts_image_dir,
                        label_dir=base_dir / "labelsTr",
                        target_dir=target_dir,
                        export_stuff=stuff,
                        ).export()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tasks', type=str, nargs='+',
                        help="Single or multiple task identifiers to process consecutively",
                        )
    parser.add_argument('-nt', '--new_tasks', type=str, nargs='+',
                        help="Rename the tasks.",
                        required=False, default=None,
                        )
    parser.add_argument("--stuff", action='store_true',
                        help="Export stuff and things classes."
                             "The final detection evaluation will be performed on things classes only.")
    parser.add_argument('-o', '--overwrites', type=str, nargs='+',
                        help="overwrites for config file",
                        required=False,
                        )

    args = parser.parse_args()
    tasks = args.tasks
    new_tasks = args.new_tasks
    ov = args.overwrites
    stuff = args.stuff
    print(f"Overwrites: {ov}")
    initialize_config_module(config_module="nndet.conf")

    if new_tasks is None:
        new_tasks = tasks

    for task, new_task in zip(tasks, new_tasks):
        task = get_task(task, name=True)

        if nnUNet_raw_data is None:
            raise RuntimeError(f"Please set `nnUNet_raw_data` for nnUNet!")
        target_dir = Path(nnUNet_raw_data) / new_task

        logger.remove()
        logger.add(sys.stdout, level="INFO")
        logger.add(target_dir / "nnunet_export.log", level="DEBUG")

        _ov = copy.deepcopy(ov) if ov is not None else []
        cfg = compose(task, "config.yaml", overrides=ov if ov is not None else [])
        print(cfg)
        run(cfg, target_dir, stuff=stuff)
