# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import copy
import functools
import inspect
import multiprocessing
import os
import pathlib
import socket
import sys
from collections.abc import MutableMapping
from contextlib import contextmanager
from pathlib import Path
from subprocess import PIPE, run
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Union

import pytorch_lightning as pl
import torch
from git import InvalidGitRepositoryError, Repo
from loguru import logger
from pytorch_lightning.callbacks import ModelSummary as _ModelSummary
from pytorch_lightning.utilities.model_summary import _format_summary_table, summarize
from tqdm import tqdm

from nndet.io.load import save_txt


class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class ModelSummary(_ModelSummary):
    def __init__(
        self,
        max_depth: int = 1,
        log_net: bool = True,
    ) -> None:
        super().__init__(max_depth=max_depth)
        self.log_net = log_net

    def on_fit_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        if not self._max_depth:
            return None

        model_summary = summarize(pl_module, max_depth=self._max_depth)
        summary_data = model_summary._get_summary_data()
        total_parameters = model_summary.total_parameters
        trainable_parameters = model_summary.trainable_parameters
        model_size = model_summary.model_size

        if trainer.is_global_zero:
            summary_table = _format_summary_table(
                total_parameters,
                trainable_parameters,
                model_size,
                *summary_data,
            )

            summary_full = f"+++ Network Summary +++ \n\n{summary_table} \n\n{pl_module}"
            Path("./network.txt").unlink(missing_ok=True)
            save_txt(summary_full, "./network")

            if self.log_net:
                logger.info(summary_full)


def deprecate(
    replacement: Optional[str] = None,
    deprecate: Optional[str] = None,
    remove: Optional[str] = None,
):
    """
    Deprecate functions and classes

    Args:
        replacement: Optional replacement of old element. if No
            replacement is provided (None) this will expect that the function
            will be removed completely.
        deprecate: Optional version from when element is deprecated.
        remove: Optional version from when element will be removed.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if inspect.isclass(func):
                func_name = func.__class__.__name__
            else:
                func_name = func.__qualname__

            time_str = "now" if deprecate is None else deprecate

            s = f"{func_name} is deprecated from {time_str}!"

            if remove is not None:
                s += f" It will be removed from nnDetection {remove}"
            if replacement is not None:
                s += f" The replacement is {replacement}."
            else:
                s += " There will be no replacement."

            logger.warning(s)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def experimental(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if inspect.isclass(func):
            func_name = func.__class__.__name__
        else:
            func_name = func.__qualname__

        logger.warning(
            f"This feature ({func_name}) is experimental! "
            "It might not implement all features or is only a simplification!"
        )
        return func(*args, **kwargs)

    return wrapper


def get_requirements():
    """
    Get all installed packages from currently active environment

    Returns:
        str: list with all requirements
    """
    command = ["pip", "list"]
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
    return {
        "hash": repo.head.commit.hexsha,
        "gitdir": repo.git_dir,
        "active_branch": repo.active_branch.name,
    }


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


def find_name(tdir: Union[str, Path], name: str, postfix: Optional[str] = None) -> Path:
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

    i = 0
    while True:
        output_dir = tdir / f"{name}{i:03d}{postfix}"
        if not output_dir.exists():
            break
        if i > 1000:
            raise RuntimeError(f"Was not able to find name for tdir {tdir} and {name}")
        i += 1
    return output_dir


def log_git(repo_path: Union[pathlib.Path, str], repo_name: str = None):
    """
    Use python logging module to log git information

    Args:
        repo_path: path to repo or file inside repository (repository is recursively searched)
    """
    try:
        git_info = get_repo_info(repo_path)
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
    cls_name = cls_name.split("'")[1]
    # split modules
    cls_split = cls_name.split(".")
    if len(cls_split) > 1:
        cls_name = cls_split[0] + "." + cls_split[-1] if package_name else cls_split[-1]
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


def host_and_env_info() -> Dict[str, Union[str, float, int]]:
    info = {}

    # host info
    info["hostname"] = socket.gethostname()
    info["job_id"] = os.getenv("LSB_JOBID", "no_id")
    try:
        info["cpu_count"] = multiprocessing.cpu_count()
    except NotImplementedError:
        info["cpu_count"] = -1
    for i in range(torch.cuda.device_count()):
        info[f"gpu{i}"] = torch.cuda.get_device_name(i)

    # env info
    info["det_num_threads"] = os.environ.get("det_num_threads", -1)
    return info
