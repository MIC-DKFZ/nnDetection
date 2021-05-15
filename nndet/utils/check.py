import functools
import os
import warnings

from nndet.io.paths import get_task
from nndet.utils.config import load_dataset_info


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


def _check_key_missing(cfg: dict, key: str, ktype=None):
    if key not in cfg:
        raise ValueError(f"Dataset information did not contain "
                        f"'{key}' key, found {list(cfg.keys())}")
    
    if ktype is not None:
        if not isinstance(cfg[key], ktype):
            raise ValueError(f"Found {key} of type {type(cfg[key])} in "
                             f"dataset information but expected type {ktype}")


def check_dataset_file(task_name: str):
    """
    Run a sequence of checks to confirm correct format of dataset information

    Args:
        task_name: task identifier to check info for
    """
    cfg = load_dataset_info(get_task(task_name))
    _check_key_missing(cfg, "task", ktype=str)
    _check_key_missing(cfg, "dim", ktype=int)
    _check_key_missing(cfg, "labels", ktype=dict)
    _check_key_missing(cfg, "modalities", ktype=dict)

    # check dim
    if dim := cfg["dim"] not in [2, 3]:
        raise ValueError(f"Found dim {dim} in dataset info but only support dim=2 or dim=3.")

    # check labels
    for key, item in cfg["labels"].items():
        if not isinstance(key, (str, int)):
            raise ValueError("Expected key of type string in dataset "
                             f"info labels but found {type(key)} : {key}")
        if not isinstance(item, (str, int)):
            raise ValueError("Expected name of type string in dataset "
                             f"info labels but found {type(item)} : {item}")
    found_classes = sorted(list(map(int, cfg["labels"].keys())))
    for ic, idx in enumerate(found_classes):
        if ic != idx:
            raise ValueError("Found wrong order of label classes in dataset info."
                             f"Found {found_classes} but expected {list(range(len(found_classes)))}")

    # check modalities
    for key, item in cfg["modalities"].items():
        if not isinstance(key, (str, int)):
            raise ValueError("Expected key of type string in dataset "
                             f"info labels but found {type(key)} : {key}")
        if not isinstance(item, (str, int)):
            raise ValueError("Expected name of type string in dataset "
                             f"info labels but found {type(item)} : {item}")
    found_mods = sorted(list(map(int, cfg["modalities"].keys())))
    for ic, idx in enumerate(found_classes):
        if ic != idx:
            raise ValueError("Found wrong order of modalities in dataset info."
                             f"Found {found_mods} but expected {list(range(len(found_mods)))}")


def check_data_and_label_splitted():
    pass
