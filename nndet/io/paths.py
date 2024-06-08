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

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

Pathlike = Union[Path, str]


def subfiles(dir_path: Path, identifier: str, join: bool) -> List[str]:
    """
    Get all paths

    Args:
        dir_path: path to directory
        join: return dir_path+file_name instead of file_name
        identifier: regular expression to select files

    Returns:
        List[str]: found paths/file names
    """
    paths = list(map(str, list(Path(os.path.expandvars(dir_path)).glob(identifier))))
    if not join:
        paths = [p.rsplit(os.path.sep, 1)[-1] for p in paths]
    return paths


def get_paths_raw_to_split(data_dir: Path, output_dir: Path,
                           subdirs: tuple = ("imagesTr", "imagesTs")) -> Tuple[
        List[Path], List[Path]]:
    """
    Search subdirs for all *.nii.gz files which need to be splitted and
    create lists with source and target paths of all files
    (target paths retain subfolders inside of output dir)

    Args:
        data_dir (str): top directory where data is located
        output_dir (str): output directory for splitted data
        subdirs (Tuple[str]): subdirectories which should be searched for data

    Returns:
        List[Path]: path to all nii files in subfolders of source directory
        List[Path]: path to respective target directory
    """
    source_files, target_dirs = [], []

    for subdir in subdirs:
        sub_output_dir = output_dir / subdir
        if not sub_output_dir.is_dir():
            sub_output_dir.mkdir(parents=True)

        sub_data_dir = data_dir / subdir
        nii_files = list(sub_data_dir.glob('*.nii.gz'))
        nii_files = list(filter(lambda x: not x.name.startswith('.'), nii_files))
        nii_files.sort()
        for n in nii_files:
            source_files.append(n)
            target_dirs.append(sub_output_dir)
    return source_files, target_dirs


def get_paths_from_splitted_dir(
    num_modalities: int,
    splitted_4d_output_dir: Path,
    test: bool = False,
    labels: bool = True,
    remove_ids: Optional[Sequence[str]] = None,
    ) -> List[List[Path]]:
    """
    Create list to all cases (data and label; label is at last position) inside splitted data dir

    Args:
        num_modalities (int): number of modalities
        splitted_4d_output_dir (Path): path to dir where 4d splitted data is located
        test: get paths from test data (if False, searches for train data)
        labels: add path to labels at last position of each case
        remove_ids: case ids which should be removed from the list. If None,
            no case ids are removed

    Returns:
        List[List[Path]]: paths to all splitted files;
            each case contains its data files and the label file is at the end
    """
    data_subdir = "imagesTs" if test else "imagesTr"
    labels_subdir = "labelsTs" if test else "labelsTr"
    training_ids = get_case_ids_from_dir(
        splitted_4d_output_dir / data_subdir,
        remove_modality=True,
        )
    if remove_ids is not None:
        training_ids = [t for t in training_ids if t not in remove_ids]

    all_cases = []
    training_ids.sort()
    for case_id in training_ids:
        case_paths = []

        for mod in range(num_modalities):
            case_paths.append(
                splitted_4d_output_dir / data_subdir / f"{case_id}_{mod:04d}.nii.gz")
        if labels:
            case_paths.append((splitted_4d_output_dir / labels_subdir) / f"{case_id}.nii.gz")
        all_cases.append(case_paths)
    return all_cases


def get_case_ids_from_dir(dir_path: Path, unique: bool = True,
                          remove_modality: bool = True, join: bool = False,
                          pattern="*.nii.gz") -> List[str]:
    """
    Get all case ids from a single folder

    Args:
        dir_path: path to folder
        unique: remove all duplicates
        remove_modality: remove the modality string from the filename
        join: append case ids to directory path
        pattern: regular expression used to select files

    Returns:
        List[str]: all case ids inside the folder
    """
    files = map(str, list(Path(dir_path).glob(pattern)))
    case_ids = [get_case_id_from_path(f, remove_modality=remove_modality) for f in files]
    if unique:
        case_ids = list(set(case_ids))
    if join:
        case_ids = [os.path.join(dir_path, c) for c in case_ids]
    return case_ids


def get_case_id_from_path(file_path: Pathlike, remove_modality: bool = True) -> str:
    """
    Get case of from path to file

    Args:
        file_path (str): path to file as string
        remove_modality (bool): remove the modality string from the filename
            (only used if file ends with .nii.gz)

    Returns:
        str: case id
    """
    file_name = str(file_path).rsplit(os.path.sep, 1)[1]
    return get_case_id_from_file(file_name, remove_modality=remove_modality)


def get_case_id_from_file(file_name: str, remove_modality: bool = True) -> str:
    """
    Cut of ".nii.gz" from file name

    Args:
        file_name (str): name of file with .nii.gz ending
        remove_modality (bool): remove the modality string from the filename

    Returns:
        str: name of file without ending
    """
    if file_name.endswith(".nii.gz"):
        file_name = file_name.rsplit(".", 2)[0]
    else:
        file_name = file_name.rsplit(".", 1)[0]

    if remove_modality:
        file_name = file_name[:-5]
    return file_name


def get_task(task_id: str, name: bool = False, models: bool = False) -> Union[Path, str]:
    """
    Resolve task name/dir

    Args:
        task_id: identifier of task.
            E.g. task dir = ../Task12_LIDC
            Possible task ids: Task12, LIDC, Task12_LIDC
        name: only return the name of the task
        models: uses model folder to look for names

    Returns:
        Union[Path, str]:
            path to data task directory if name is False
            name of task if name is True
    """
    if models:
        t = os.getenv("det_models")
    else:
        t = os.getenv("det_data")
    if t is None:
        raise ValueError("Framework not configured correctly! "
                         "Please set `det_data` and `det_models` as environment variables!")
    det_data = Path(t)
    all_tasks = [d.stem for d in det_data.iterdir() if d.is_dir() and "Task" in d.name]

    if task_id.startswith("Task"):
        task_id = task_id[4:]
    all_tasks = [tn[4:] for tn in all_tasks]

    task_options_exact = [d for d in all_tasks if task_id in d]
    task_number_id = [tn for tn in all_tasks if tn.split('_', 1)[0] == task_id]
    task_name_id = [tn for tn in all_tasks if tn.split('_', 1)[1] == task_id]

    if len(task_options_exact) == 1:
        result = det_data / f"Task{task_options_exact[0]}"
    elif len(task_number_id) == 1:
        result = det_data / f"Task{task_number_id[0]}"
    elif len(task_name_id) == 1:
        result = det_data / f"Task{task_name_id[0]}"
    else:
        raise ValueError(f"Did not find task id {task_id}."
                         f"Options are: {all_tasks}")
    if name:
        result = result.stem
    return result


def get_training_dir(model_dir: Pathlike, fold: int) -> Path:
    """
    Find training dir from a specific model dir

    Args:
        model_dir: path to model dir e.g. ../Task12_LIDC/RetinaUNetV0
        fold: fold to look for. if -1 look for consolidated dir

    Returns:
        Path: path to training dir
    """
    model_dir = Path(model_dir)
    identifier = f"fold{fold}" if fold != -1 else "consolidated"
    candidates = [p for p in model_dir.iterdir() if p.is_dir() and identifier in p.stem]
    if len(candidates) == 1:
        return candidates[0]
    else:
        raise ValueError(f"Found wrong number of training dirs {candidates} in {model_dir}")
