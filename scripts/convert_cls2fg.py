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

import argparse
import shutil
import sys
from pathlib import Path

from hydra import initialize_config_module
from loguru import logger

from nndet.io import get_task, load_json, save_json
from nndet.utils.config import compose, load_dataset_info
from nndet.utils.check import env_guard


def convert_raw(task, overwrite, ov):
    task_name_full = get_task(task, name=True)
    task_num, task_name = task_name_full[4:].split('_', 1)
    new_task_name_full = f"Task{task_num}FG_{task_name}"

    cfg = compose(task, "config.yaml", overrides=ov if ov is not None else [])
    print(cfg)

    source_splitted_dir = Path(cfg["host"]["splitted_4d_output_dir"])
    target_splitted_dir = Path(str(source_splitted_dir).replace(task_name_full, new_task_name_full))
    if target_splitted_dir.is_dir() and overwrite:
        shutil.rmtree(target_splitted_dir)
    target_splitted_dir.mkdir(parents=True)

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(target_splitted_dir.parent / "convert_cls2fg.log", level="DEBUG")

    # update dataset_info
    source_data_info = Path(cfg["host"]["data_dir"])
    data_info = load_dataset_info(source_data_info)
    data_info.pop("labels")
    data_info["labels"] = {"0": "fg"}
    data_info["task"] = new_task_name_full
    save_json(data_info, target_splitted_dir.parent / "dataset.json", indent=4)

    for postfix in ["Tr", "Ts"]:
        source_image_dir = source_splitted_dir / f"images{postfix}"
        source_label_dir = source_splitted_dir / f"labels{postfix}"

        if not source_image_dir.is_dir():
            logger.info(f"{source_image_dir} is not a dir. Skipping it.")
            continue

        # copy images and labels
        shutil.copytree(source_image_dir, target_splitted_dir / f"images{postfix}")
        shutil.copytree(source_label_dir, target_splitted_dir / f"labels{postfix}")

        # remap properties file to foreground class
        target_label_dir = target_splitted_dir / f"labels{postfix}"
        for f in [l for l in target_label_dir.glob("*.json")]:
            props = load_json(f)
            props["instances"] = {key: 0 for key in props["instances"].keys()}
            save_json(props, f)


@env_guard
def main():
    """
    Convert raw splitted data with class sensitive annotations into
    a new dataset which only distinguishes fg and bg
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('tasks', type=str, nargs='+',
                        help="Single or multiple task identifiers to process consecutively",
                        )
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('-o', '--overwrites', type=str, nargs='+',
                        help="overwrites for config file",
                        required=False)
    args = parser.parse_args()
    tasks = args.tasks
    ov = args.overwrites
    overwrite = args.overwrite
    initialize_config_module(config_module="nndet.conf")

    for task in tasks:
        convert_raw(task, overwrite, ov)


if __name__ == '__main__':
    main()
