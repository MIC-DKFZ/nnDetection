from pathlib import Path
from typing import Dict
from unittest.mock import patch

import numpy as np
import pytest
import SimpleITK as sitk

from nndet.io import save_json
from nndet.utils.check import (
    _check_dataset_file,
    _check_instances_json,
    _check_itk_params,
    check_dataset_file,
    check_torch_version,
)


@pytest.fixture
def dataset_info():
    return {
        "task": "Task000D3_Example",
        "name": "Example",  # [Optional]
        "dim": 3,  # number of spatial dimensions of the data
        # Note: need to use integer value which is defined below of target class!
        "target_class": 1,  # [Optional] define class of interest for patient level evaluations
        "test_labels": True,  # manually splitted test set
        # classes of data set; need to start at 0
        "labels": {
            "0": "Square",
            "1": "SquareHole",
        },
        "modalities": {
            "0": "CT",
            "1": "CT",
        },  # modalities of data set; need to start at 0
        "annotation_style": "seg",
    }


@pytest.fixture
def label_json():
    return {
        "instances": {
            "1": 1,
            "2": 1,
            "3": 0,
        }
    }


def test_check_dataset_file_env(tmpdir, monkeypatch, dataset_info):
    monkeypatch.setenv("det_data", str(tmpdir))

    task_dir = Path(tmpdir) / dataset_info["task"]
    task_dir.mkdir()

    save_json(dataset_info, task_dir / "dataset.json")
    check_dataset_file(dataset_info["task"])


def test_check_dataset_file(dataset_info):
    _check_dataset_file(dataset_info)


@pytest.mark.parametrize("key", ["task", "dim", "labels", "modalities"])
def test_check_dataset_file_missing_key(
    key: str,
    dataset_info: Dict,
):
    dataset_info.pop(key)

    with pytest.raises(ValueError):
        _check_dataset_file(dataset_info)


def test_check_dataset_file_wrong_type_labels(dataset_info: Dict):
    dataset_info["labels"]["1"] = 2

    with pytest.raises(ValueError):
        _check_dataset_file(dataset_info)


def test_check_dataset_file_wrong_type_mods(dataset_info: Dict):
    dataset_info["modalities"]["0"] = 2

    with pytest.raises(ValueError):
        _check_dataset_file(dataset_info)


@pytest.mark.parametrize("key", ["name", "target_class", "test_labels"])
def test_check_dataset_file_missing_optional_key(
    key: str,
    dataset_info: Dict,
):
    dataset_info.pop(key)
    _check_dataset_file(dataset_info)


@pytest.mark.parametrize("key", ["labels", "modalities"])
def test_check_key_starts_at_zero(key, dataset_info):
    dataset_info[key].pop("0")

    with pytest.raises(ValueError):
        _check_dataset_file(dataset_info)


@pytest.mark.parametrize("key", ["labels", "modalities"])
def test_check_key_consecutive(key, dataset_info):
    r = dataset_info[key].pop("1")
    dataset_info[key]["2"] = r

    with pytest.raises(ValueError):
        _check_dataset_file(dataset_info)


def test_check_dataset_file_wrong_dim(dataset_info: Dict):
    dataset_info["dim"] = 1

    with pytest.raises(ValueError):
        _check_dataset_file(dataset_info)


@pytest.mark.parametrize("tc", ["Square", 3])
def test_check_dataset_file_wrong_target_class(dataset_info: Dict, tc):
    dataset_info["target_class"] = tc

    with pytest.raises(ValueError):
        _check_dataset_file(dataset_info)


def test_check_instances_json(label_json):
    _check_instances_json(label_json, None, ["0", "1"])


def test_check_instances_json_start_at_one(label_json):
    label_json["instances"]["0"] = 1

    with pytest.raises(ValueError):
        _check_instances_json(label_json, None, ["0", "1"])


def test_check_instances_json_consecutive(label_json):
    label_json["instances"].pop("2")

    with pytest.raises(ValueError):
        _check_instances_json(label_json, None, ["0", "1"])


def test_check_instances_json_unknown_class(label_json):
    label_json["instances"]["3"] = 3

    with pytest.raises(ValueError):
        _check_instances_json(label_json, None, ["0", "1"])


def test_check_itk_params():
    ref = sitk.GetImageFromArray(np.zeros((10, 10, 10)))
    ref2 = sitk.GetImageFromArray(np.zeros((10, 10, 10)))

    _check_itk_params([ref, ref2], [None, None])


def test_check_itk_params_dimension():
    ref = sitk.GetImageFromArray(np.zeros((10, 10, 10)))
    ref2 = sitk.GetImageFromArray(np.zeros((10, 10)))

    with pytest.raises(ValueError):
        _check_itk_params([ref, ref2], [None, None])


def test_check_itk_params_size():
    ref = sitk.GetImageFromArray(np.zeros((10, 10, 10)))
    ref2 = sitk.GetImageFromArray(np.zeros((10, 10, 11)))

    with pytest.raises(ValueError):
        _check_itk_params([ref, ref2], [None, None])


def test_check_itk_params_direction():
    ref = sitk.GetImageFromArray(np.zeros((10, 10, 10)))
    ref2 = sitk.GetImageFromArray(np.zeros((10, 10, 10)))

    d = list(ref.GetDirection())
    d[0] = 2.0
    ref2.SetDirection(tuple(d))

    with pytest.raises(ValueError):
        _check_itk_params([ref, ref2], [None, None])


def test_check_itk_params_spacing():
    ref = sitk.GetImageFromArray(np.zeros((10, 10, 10)))
    ref2 = sitk.GetImageFromArray(np.zeros((10, 10, 10)))

    d = list(ref.GetSpacing())
    d[0] = 2.0
    ref2.SetSpacing(tuple(d))

    with pytest.raises(ValueError):
        _check_itk_params([ref, ref2], [None, None])


def test_check_itk_params_origin():
    ref = sitk.GetImageFromArray(np.zeros((10, 10, 10)))
    ref2 = sitk.GetImageFromArray(np.zeros((10, 10, 10)))

    d = list(ref.GetOrigin())
    d[0] = 2.0
    ref2.SetOrigin(tuple(d))

    with pytest.raises(ValueError):
        _check_itk_params([ref, ref2], [None, None])


@patch("torch.__version__", "2.0.0+cu118")
def test_check_torch_major_true():
    assert check_torch_version(major_version=2)


@patch("torch.__version__", "2.0.0+cu118")
def test_check_torch_minor_true():
    assert check_torch_version(major_version=2, minor_version=0)


@patch("torch.__version__", "1.12.1+cu116")
def test_check_torch_major_false():
    assert not check_torch_version(major_version=2)


@patch("torch.__version__", "1.12.1+cu116")
def test_check_torch_minor_false():
    assert not check_torch_version(major_version=1, minor_version=13)
