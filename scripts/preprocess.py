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
import os
import copy
import sys
import traceback

import numpy as np

from loguru import logger
from itertools import repeat
from typing import Dict, Sequence, Tuple, List
from pathlib import Path
from multiprocessing import Pool
from hydra import initialize_config_module
from omegaconf import OmegaConf

from nndet.utils.config import compose
from nndet.utils.check import env_guard
from nndet.planning import DatasetAnalyzer
from nndet.planning import PLANNER_REGISTRY
from nndet.planning.experiment.utils import create_labels
from nndet.planning.properties.registry import medical_instance_props
from nndet.io.load import load_pickle, load_npz_looped
from nndet.io.paths import get_paths_raw_to_split, get_paths_from_splitted_dir, subfiles, get_case_id_from_path
from nndet.preprocessing import ImageCropper
from nndet.utils.check import check_dataset_file, check_data_and_label_splitted


def run_cropping_and_convert(cropped_output_dir: Path,
                             splitted_4d_output_dir: Path,
                             data_info: dict,
                             overwrite: bool,
                             num_processes: int,
                             ):
    """
    First preparation step data:
        - stack data and segementation to a single sample (segmentation is the last channel)
        - save data as npz (format: case_id.npz)
        - save additional properties as pkl file (format: case_id.pkl)
        - crop data to nonzeor region; crop segmentation; fill segmentation with -1 where in nonzero regions

    Args:
        cropped_output_dir (Path): path to directory where cropped images should be saved
        splitted_4d_output_dir (Path): path to splitted data
        data_info: information about data set (here `modalities` is needed)
        overwrite (bool): overwrite existing cropped data
        num_processes (int): number of processes used to crop image data
    """
    num_modalities = len(data_info["modalities"].keys())

    if overwrite and cropped_output_dir.is_dir():
        shutil.rmtree(str(cropped_output_dir))
    if not cropped_output_dir.is_dir():
        cropped_output_dir.mkdir(parents=True)

    case_files = get_paths_from_splitted_dir(num_modalities, splitted_4d_output_dir)

    logger.info(f"Running cropping with overwrite {overwrite}.")
    imgcrop = ImageCropper(num_processes, cropped_output_dir)
    imgcrop.run_cropping(case_files, overwrite_existing=overwrite)

    case_ids_failed, result_check = run_check(cropped_output_dir / "imagesTr",
                                              remove=True,
                                              processes=num_processes,
                                              keys=("data",)
                                              )
    if not result_check:
        logger.warning(
            f"Crop check failed: There are corrupted files!!!! {case_ids_failed}"
            f"Try to crop corrupted files again.",
        )
        imgcrop = ImageCropper(0, cropped_output_dir)
        imgcrop.run_cropping(case_files, overwrite_existing=False)
        case_ids_failed, result_check = run_check(cropped_output_dir / "imagesTr",
                                                  remove=False,
                                                  processes=num_processes,
                                                  keys=("data",)
                                                  )
        if not result_check:
            logger.error(f"Found corrupted files: {case_ids_failed}.")
            raise RuntimeError("Corrupted files")
    else:
        logger.info(f"Crop check successful: Loading check completed")


def run_dataset_analysis(cropped_output_dir: Path,
                         preprocessed_output_dir: Path,
                         data_info: dict,
                         num_processes: int,
                         intensity_properties: bool = True,
                         overwrite: bool = True,
                         ):
    """
    Analyse dataset

    Args:
        cropped_output_dir: path to base cropped dir
        preprocessed_output_dir: path to base preprocessed output dir
        data_info: additional information about dataset (`modalities` and `labels` needed)
        num_processes: number of processes to use
        intensity_properties: analyze intensity values of foreground
        overwrite: overwrite existing properties
    """
    analyzer = DatasetAnalyzer(
        cropped_output_dir,
        preprocessed_output_dir=preprocessed_output_dir,
        data_info=data_info,
        num_processes=num_processes,
        overwrite=overwrite,
        )
    properties = medical_instance_props(intensity_properties=intensity_properties)
    _ = analyzer.analyze_dataset(properties)


def run_planning_and_process(
    splitted_4d_output_dir: Path,
    cropped_output_dir: Path,
    preprocessed_output_dir: Path,
    planner_name: str,
    dim: int,
    model_name: str,
    model_cfg: Dict,
    num_processes: int,
    run_preprocessing: bool = True,
    ):
    """
    Run planning and preprocessing

    Args:
        splitted_4d_output_dir: base dir of splitted data
        cropped_output_dir: base dir of cropped data
        preprocessed_output_dir: base dir of preprocessed data
        planner_name: planner name
        dim: number of spatial dimensions
        model_name: name of model to run planning for
        model_cfg: hyperparameters of model (used during planning to
            instantiate model)
        num_processes: number of processes to use for preprocessing
        run_preprocessing: Preprocess and check data. Defaults to True.
    """
    planner_cls = PLANNER_REGISTRY.get(planner_name)
    planner = planner_cls(
        preprocessed_output_dir=preprocessed_output_dir
    )
    plan_identifiers = planner.plan_experiment(
        model_name=model_name,
        model_cfg=model_cfg,
    )
    if run_preprocessing:
        for plan_id in plan_identifiers:
            plan = load_pickle(preprocessed_output_dir / plan_id)
            planner.run_preprocessing(
                cropped_data_dir=cropped_output_dir / "imagesTr",
                plan=plan,
                num_processes=num_processes,
                )
            case_ids_failed, result_check = run_check(
                data_dir=preprocessed_output_dir / plan["data_identifier"] / "imagesTr",
                remove=True,
                processes=num_processes
            )

            # delete and rerun corrupted cases
            if not result_check:
                logger.warning(f"{plan_id} check failed: There are corrupted files {case_ids_failed}!!!!"
                                f"Running preprocessing of those cases without multiprocessing.")
                planner.run_preprocessing(
                    cropped_data_dir=cropped_output_dir / "imagesTr",
                    plan=plan,
                    num_processes=0,
                )
                case_ids_failed, result_check = run_check(
                    data_dir=preprocessed_output_dir / plan["data_identifier"] / "imagesTr",
                    remove=False,
                    processes=0
                )
                if not result_check:
                    logger.error(f"Could not fix corrupted files {case_ids_failed}!")
                    raise RuntimeError("Found corrupted files, check logs!")
                else:
                    logger.info("Fixed corrupted files.")
            else:
                logger.info(f"{plan_id} check successful: Loading check completed")

    if run_preprocessing:
        create_labels(
            preprocessed_output_dir=preprocessed_output_dir,
            source_dir=splitted_4d_output_dir,
            num_processes=num_processes,
        )


def run_check(data_dir: Path,
              remove: bool = False,
              processes: int = 8,
              keys: Sequence[str] = ("data", "seg"),
              ) -> Tuple[List[str], bool]:
    """
    Check if files from preprocessed dir are loadable

    Args:
        data_dir (Path): path to preprocessed data
        remove (bool, optional): if loading fails the file is the npz and pkl
            file are removed automatically. Defaults to False.
        processes (int, optional): number of processes to use. If
            0 processes are specified it uses a normal for loop. Defaults to 8.
        keys: keys to load and check

    Returns:
        True if all cases were loadable, False otherwise
    """
    cases_npz = list(data_dir.glob("*.npz"))
    cases_npz.sort()
    cases_pkl = [case.parent / f"{(case.name).rsplit('.', 1)[0]}.pkl"
                 for case in cases_npz]

    if processes == 0:
        result = [check_case(case_npz, case_pkl, remove=remove, keys=keys)
                  for case_npz, case_pkl in zip(cases_npz, cases_pkl)]
    else:
        with Pool(processes=processes) as p:
            result = p.starmap(check_case,
                               zip(cases_npz, cases_pkl, repeat(remove), repeat(keys)))
    failed_cases = [fc[0] for fc in result if not fc[1]]
    logger.info(f"Checked {len(result)} cases in {data_dir}")
    return failed_cases, len(failed_cases) == 0


def check_case(case_npz: Path,
               case_pkl: Path = None,
               remove: bool = False,
               keys: Sequence[str] = ("data", "seg"),
               ) -> Tuple[str, bool]:
    """
    Check if a single cases loadable

    Args:
        case_npz (Path): path to npz file
        case_pkl (Path, optional): path to pkl file. Defaults to None.
        remove (bool, optional): if loading fails the file is the npz and pkl
            file are removed automatically. Defaults to False.

    Returns:
        str: case id
        bool: true if case was loaded correctly, false otherwise
    """
    logger.info(f"Checking {case_npz}")
    case_id = get_case_id_from_path(case_npz, remove_modality=False)
    try:
        case_dict = load_npz_looped(str(case_npz), keys=keys, num_tries=3)
        if "seg" in keys and case_pkl is not None:
            properties = load_pickle(case_pkl)
            seg = case_dict["seg"]
            seg_instances = np.unique(seg)  # automatically sorted
            seg_instances = seg_instances[seg_instances > 0]
            
            instances_properties = properties["instances"].keys()
            props_instances = np.sort(np.array(list(map(int, instances_properties))))
            
            if (len(seg_instances) != len(props_instances)) or any(seg_instances != props_instances):
                logger.warning(f"Inconsistent instances {case_npz} from "
                                f"properties {props_instances} from seg {seg_instances}. "
                                f"Very small instances can get lost in resampling "
                                f"but larger instances should not disappear!")       
            for i in seg_instances:
                if str(i) not in instances_properties:
                    raise RuntimeError(f"Found instance {seg_instances} in segmentation "
                                       f"which is not in properties {instances_properties}."
                                       f"Delete labels manually and rerun prepare label!")
    except Exception as e:
        logger.error(f"Failed to load {case_npz} with {e}")
        logger.error(f"{traceback.format_exc()}")
        if remove:
            os.remove(case_npz)
            if case_pkl is not None:
                os.remove(case_pkl)
        return case_id, False
    return case_id, True


def run(cfg,
        num_processes: int,
        num_processes_preprocessing: int,
        ):
    """
    Python interface for script

    Args:
        cfg: dict with config
        instances_from_seg: convert semantic segmentation to instance segmentation
    """
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(Path(cfg["host"]["data_dir"]) / "logging.log", level="DEBUG")
    data_info = cfg["data"]

    if cfg["prep"]["crop"]:
        # crop data to nonzero area
        run_cropping_and_convert(cropped_output_dir=Path(cfg["host"]["cropped_output_dir"]),
                                 splitted_4d_output_dir=Path(cfg["host"]["splitted_4d_output_dir"]),
                                 data_info=data_info,
                                 overwrite=cfg["prep"]["overwrite"],
                                 num_processes=num_processes,
                                 )

    if cfg["prep"]["analyze"]:
        # compute statistics over data and segmentation(e.g. physical volume of individual classes)
        run_dataset_analysis(cropped_output_dir=Path(cfg["host"]["cropped_output_dir"]),
                             preprocessed_output_dir=Path(cfg["host"]["preprocessed_output_dir"]),
                             data_info=data_info,
                             num_processes=num_processes,
                             intensity_properties=True,
                             overwrite=cfg["prep"]["overwrite"],
                             )

    if cfg["prep"]["plan"] or cfg["prep"]["process"]:
        # plan future training
        run_planning_and_process(
            splitted_4d_output_dir=Path(cfg["host"]["splitted_4d_output_dir"]),
            cropped_output_dir=Path(cfg["host"]["cropped_output_dir"]),
            preprocessed_output_dir=Path(cfg["host"]["preprocessed_output_dir"]),
            planner_name=cfg["planner"],
            dim=data_info["dim"],
            model_name=cfg["module"],
            model_cfg=cfg["model_cfg"],
            num_processes=num_processes_preprocessing,
            run_preprocessing=cfg["prep"]["process"],
        )


@env_guard
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tasks', type=str, nargs='+',
                        help="Single or multiple task identifiers to process consecutively",
                        )
    parser.add_argument('-o', '--overwrites', type=str, nargs='+',
                        help="overwrites for config file", default=[],
                        required=False)
    parser.add_argument('--full_check',
                        help="Run a full check of the data.",
                        action='store_true',
                        )
    parser.add_argument('--no_check',
                        help="Skip basic check.",
                        action='store_true',
                        )
    parser.add_argument('-np', '--num_processes',
                        type=int, default=4, required=False,
                        help="Number of processes to use for croppping.",
                        )
    parser.add_argument('-npp', '--num_processes_preprocessing',
                        type=int, default=3, required=False,
                        help="Number of processes to use for resampling.",
                        )
    args = parser.parse_args()
    tasks = args.tasks
    ov = args.overwrites
    full_check = args.full_check
    no_check = args.no_check
    num_processes = args.num_processes
    num_processes_preprocessing = args.num_processes_preprocessing

    initialize_config_module(config_module="nndet.conf")
    # perform preprocessing checks first
    if not no_check:
        for task in tasks:
            _ov = copy.deepcopy(ov) if ov is not None else []
            cfg = compose(task, "config.yaml", overrides=_ov)
            check_dataset_file(cfg["task"])
            check_data_and_label_splitted(
                cfg["task"],
                test=False,
                labels=True,
                full_check=full_check,
                )
            if cfg["data"]["test_labels"]:
                check_data_and_label_splitted(
                    cfg["task"],
                    test=True,
                    labels=True,
                    full_check=full_check,
                    )

    # start preprocessing
    for task in tasks:
        _ov = copy.deepcopy(ov) if ov is not None else []
        cfg = compose(task, "config.yaml", overrides=_ov)
        run(OmegaConf.to_container(cfg, resolve=True),
            num_processes=num_processes,
            num_processes_preprocessing=num_processes_preprocessing,
            )


if __name__ == '__main__':
    main()
