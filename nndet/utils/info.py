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
import copy
import pathlib
import warnings
import functools

from collections.abc import MutableMapping
from subprocess import PIPE, run
from omegaconf.omegaconf import OmegaConf

from tqdm import tqdm
from typing import Mapping, Sequence, Union, Callable, Any, Iterable
from loguru import logger
from contextlib import contextmanager
from typing import Union, Optional
from pathlib import Path
from git import Repo, InvalidGitRepositoryError


def env_guard(func):
    """
    Contextmanager to check nnDetection environment variables
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # we use print here because logging might not be initialized yet and
        # this is intended as a user warning.
        
        # det_data
        if os.environ.get("det_data", None) is None:
            raise RuntimeError(
                "'det_data' environment variable not set. "
                "Please refer to the installation instructions. "
                )

        # det_models
        if os.environ.get("det_models", None) is None:
            raise RuntimeError(
                "'det_models' environment variable not set. "
                "Please refer to the installation instructions. "
                )

        # OMP_NUM_THREADS
        if os.environ.get("OMP_NUM_THREADS", None) is None:
            raise RuntimeError(
                "'OMP_NUM_THREADS' environment variable not set. "
                "Please refer to the installation instructions. "
                )

        # det_num_threads
        if os.environ.get("det_num_threads", None) is None:
            warnings.warn(
                "Warning: 'det_num_threads' environment variable not set. "
                "Please read installation instructions again. "
                "Training will not work properly.")

        # det_verbose
        if os.environ.get("det_verbose", None) is None:
            print("'det_verbose' environment variable not set. "
                  "Continue in verbose mode.")

        return func(*args, **kwargs)
    return wrapper


def get_requirements():
    """
    Get all installed packages from currently active environment

    Returns:
        str: list with all requirements
    """
    command = ['pip', 'list']
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    assert not result.stderr, "stderr not empty"
    return result.stdout


def write_requirements_to_file(path: Union[str, Path]) -> None:
    """
    Write all installed packages from currently active environment to file

    Args:
        path (str): path to file (including file name and extension)
    """
    with open(path, "w+") as f:
        f.write(get_requirements())


def get_repo_info(path: Union[str, Path]):
    """
    Parse repository information from path

    Args:
        path (str): path to repo. If path is not a repository it
        searches parent folders for a repository

    Returns:
        dict: contains the current hash, gitdir and active branch
    """
    def find_repo(findpath):
        p = Path(findpath).absolute()
        for p in [p, *p.parents]:
            try:
                repo = Repo(p)
                break
            except InvalidGitRepositoryError:
                pass
        else:
            raise InvalidGitRepositoryError
        return repo
    repo = find_repo(path)
    return {"hash": repo.head.commit.hexsha,
            "gitdir": repo.git_dir,
            "active_branch": repo.active_branch.name}


def maybe_verbose_iterable(data: Iterable, **kwargs) -> Iterable:
    """
    If verbose flag of nndet is enabled, uses tqdm to create a 
    progress bar

    Args:
        data: iterable to wrap
        **kwargs: keyword arguments passed to tqdm

    Returns:
        Iterable: maybe iterable with progress bar atteched to it
    """
    if bool(int(os.getenv("det_verbose", 1))):
        return tqdm(data, **kwargs)
    else:
        return data


def find_name(tdir: Union[str, Path], name: str,
              postfix: Optional[str] = None) -> Path:
    """
    Generates non exisitng names for files and dirs by adding a counter to
    the end

    Args:
        tdir: target directory where name should be determined for
        name: base name for string
        postfix: postfix for name+counter. Defaults to None.

    Raises:
        RuntimeError: this function only works up to the counter of 1000

    Returns:
        Path: path to generated item
    """
    if not isinstance(tdir, Path):
        tdir = Path(tdir)
    if not tdir.is_dir():
        tdir.mkdir(parents=True)
    if postfix is None:
        postfix = ""

    i=0
    while True:
        output_dir = tdir / f"{name}{i:03d}{postfix}"
        if not output_dir.exists():
            break
        if i > 1000:
            raise RuntimeError(f"Was not able to find name for tdir {tdir} and {name}")
        i += 1
    return output_dir


def log_experiment(cfg, plans: dict, trainer_class: Callable, stage: int,
                   repo_path: Union[pathlib.Path, str]):
    """
    Use python logging module to log important settings

    Args:
        cfg:
            config file
        plans (dict):
            current plan for training
        trainer_class (callable):
            class of current trainer
        stage (int):
            stage
        repo_path (Union[pathlib.Path, str]):
            path to repo/or file inside a repository to parse additional information
    """
    logger.info("##################### Experiment Logging Start ##########################")
    logger.info(f"Running in {cfg.mode.mode} mode.")
    logger.info(f"I am running the following nnUNet: {cfg.exp.network}")
    logger.info(f"My trainer class is: {trainer_class}")
    logger.info("For that I will be using the following configuration:")
    log_plan(plans)
    logger.info(f"I am using stage {stage} from these plans")
    logger.info(f"\nI am using data from this folder: {os.path.join(cfg.exp.dataset_dir, plans['data_identifier'])}")
    logger.info("##################### Experiment Logging End ##########################")
    log_git(repo_path)


def log_plan(plans):
    """
    Use python logging module to log current settings

    Args:
        plans (dict):
            current plan for training
    """
    logger.info("num_classes:  {}".format(plans['num_classes']))
    logger.info("modalities:  {}".format(plans['modalities']))
    logger.info("use_mask_for_norm {}".format(plans['use_mask_for_norm']))
    logger.info("keep_only_largest_region {}".format(plans['keep_only_largest_region']))
    logger.info("min_region_size_per_class {}".format(plans['min_region_size_per_class']))
    logger.info("min_size_per_class {}".format(plans['min_size_per_class']))
    logger.info("normalization_schemes {}".format(plans['normalization_schemes']))
    logger.info("stages...\n")


def log_git(repo_path: Union[pathlib.Path, str], repo_name: str = None):
    """
    Use python logging module to log git information

    Args:
        repo_path (Union[pathlib.Path, str]): path to repo or file inside repository (repository is recursively searched)
    """
    try:
        git_info = get_repo_info(repo_path)
        # logger.info("##################### GIT INFO Start ##########################")

        # if repo_name is not None:
        #     logger.info(f"Repository name: {repo_name}")

        # for key, item in git_info.items():
        #     logger.info(f"{key}: {item}")
        # logger.info("##################### GIT INFO End ##########################")
        return git_info
    except Exception:
        logger.error("Was not able to read git information, trying to continue without.")
        return {}


def get_cls_name(obj: Any, package_name: bool = True) -> str:
    """
    Get name of class from object

    Args:
        obj (Any): any object
        package_name (bool): append package origin at the beginning

    Returns:
        str: name of class
    """
    cls_name = str(obj.__class__)
    # remove class prefix
    cls_name = cls_name.split('\'')[1]
    # split modules
    cls_split = cls_name.split('.')
    if len(cls_split) > 1:
        cls_name = cls_split[0] + '.' + cls_split[-1] if package_name else cls_split[-1]
    else:
        cls_name = cls_split[0]
    return cls_name


def log_error(fn: Callable) -> Any:
    """
    Log error messages in hydra log when they occur

    Args:
        fn: function to wrap

    Returns:
        Any
    """
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.error(str(e))
            raise e
    return wrapper


@contextmanager
def file_logger(path: Union[str, Path], level: str = "DEBUG", overwrite: bool = True):
    """
    context manager to automatically clean up file logger
    
    Args:
        path: path to output file
        level: logging level. Defaults to "Debug".
    
    Yields:
        None
    """
    path = Path(path)
    if overwrite and path.is_file():
        os.remove(path)
    logger_id = logger.add(path, level=level)
    try:
        yield None
    finally:
        logger.remove(logger_id)


def create_debug_plan(plan: dict) -> str:
    _plan = copy.deepcopy(plan)
    _plan.pop("dataset_properties", None)
    _plan.pop("original_spacings", None)
    _plan.pop("original_sizes", None)
    return stringify_nested_dict(_plan)


def stringify_nested_dict(data: dict):
    if isinstance(data, dict):
        return {str(key): stringify_nested_dict(item) for key, item in data.items()}
    elif isinstance(data, (list, tuple)):
        return [stringify_nested_dict(item) for item in data]
    else:
        return str(data)


def flatten_mapping(
    nested_mapping: Mapping,
    sep: str = ".",
    ) -> Mapping[str, Any]:
    _mapping = {}
    for key, item in nested_mapping.items():
        if isinstance(item, MutableMapping):
            for _key, _item in flatten_mapping(item, sep=sep).items():
                _mapping[str(key) + sep + str(_key)] = _item
        else:
            _mapping[str(key)] = item
    return _mapping
