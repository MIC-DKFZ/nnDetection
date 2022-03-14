import math
from abc import abstractmethod
from unittest.mock import patch

import numpy as np
import pytest

import nndet
from nndet.io.datamodule.mixins.select import (
    ObjectBalancedSelectionMixin,
    PatientBalancedSelectionMixin,
    RandomSelectionMixin,
)
from nndet.utils.enums import SelectionMode


class DataProvider:
    def __init__(self, data) -> None:
        self._data = data
        self.batch_size = 4
        self.oversample_foreground_percent = 0.5
        self.force_bg_case = False
        self.selection_mode = SelectionMode.UNIFORM

    def side_effect(self, key):
        return self._data[key]


@pytest.fixture
def no_instances_provider():
    data = {
        "o1": {"boxes_file": "o1", "instances": [], "labels": []},
        "o2": {"boxes_file": "o2", "instances": [], "labels": []},
        "o3": {"boxes_file": "o3", "instances": [], "labels": []},
    }
    return DataProvider(data)


@pytest.fixture
def filled_instances_provider():
    data = {
        "o1": {"boxes_file": "o1", "instances": [], "labels": []},
        "o2": {"boxes_file": "o2", "instances": [], "labels": []},
        "o3": {"boxes_file": "o3", "instances": [], "labels": []},
        "o4": {"boxes_file": "o4", "instances": [1], "labels": [0]},
        "o5": {"boxes_file": "o5", "instances": [1, 2], "labels": [1, 1]},
        "o6": {"boxes_file": "o6", "instances": [1, 2, 3], "labels": [0, 1, 0]},
    }
    return DataProvider(data)


@pytest.fixture
def unbal_provider():
    data = {
        "o1": {"boxes_file": "o1", "instances": [1, 2], "labels": [1, 1]},
        "o2": {"boxes_file": "o2", "instances": [1, 2], "labels": [1, 1]},
        "o3": {"boxes_file": "o3", "instances": [1, 2], "labels": [1, 1]},
        "o4": {"boxes_file": "o4", "instances": [1], "labels": [0]},
        "o5": {"boxes_file": "o5", "instances": [1, 2], "labels": [1, 1]},
        "o6": {"boxes_file": "o6", "instances": [1, 2, 3], "labels": [0, 1, 0]},
    }
    return DataProvider(data)


@pytest.fixture(scope="function")
def unbal_provider_large(request):
    rng = np.random.default_rng(seed=request.param)

    data = {}
    for i in range(1000):
        _n = f"o{i}"
        _num = rng.integers(0, 100, size=1)
        _labels = rng.choice([0, 1], size=_num, replace=True, p=[0.9, 0.1]).tolist()
        _instances = list(range(1, len(_labels) + 1))
        data[_n] = {"boxes_file": _n, "instances": _instances, "labels": _labels}
    return DataProvider(data)


class TestRandomSelection:
    select_iterations = 20
    select_iterations_full = 500

    @patch("nndet.io.datamodule.mixins.select.load_pickle")
    def test_build_cache_no_instances(self, mock_load, no_instances_provider):
        mock_load.side_effect = no_instances_provider.side_effect
        cache = RandomSelectionMixin.build_cache(no_instances_provider)
        mock_load.assert_called()

        assert cache["case"] == ["o1", "o2", "o3"]
        assert cache["bg"] == ["o1", "o2", "o3"]
        assert cache["instances"] == []

    @patch("nndet.io.datamodule.mixins.select.load_pickle")
    def test_build_cache_filled_instances(self, mock_load, filled_instances_provider):
        mock_load.side_effect = filled_instances_provider.side_effect
        cache = RandomSelectionMixin.build_cache(filled_instances_provider)
        mock_load.assert_called()

        assert cache["case"] == ["o1", "o2", "o3", "o4", "o5", "o6"]
        assert cache["bg"] == ["o1", "o2", "o3"]
        assert cache["instances"] == [
            ("o4", 1),
            ("o5", 1),
            ("o5", 2),
            ("o6", 1),
            ("o6", 2),
            ("o6", 3),
        ]

    @pytest.mark.parametrize("force_bg_case", [True, False])
    @patch("nndet.io.datamodule.mixins.select.load_pickle")
    def test_select(self, mock_load, filled_instances_provider, force_bg_case: bool):
        mock_load.side_effect = filled_instances_provider.side_effect
        filled_instances_provider.cache = RandomSelectionMixin.build_cache(
            filled_instances_provider
        )
        filled_instances_provider.force_bg_case = force_bg_case

        for _ in range(self.select_iterations):
            cases, instance_ids = RandomSelectionMixin.select(filled_instances_provider)
            assert (np.array(instance_ids) > 0).sum() == 2
            print(cases)
            if force_bg_case:
                assert all(
                    [
                        c in ["o1", "o2", "o3"]
                        for i, c in enumerate(cases)
                        if instance_ids[i] < 0
                    ]
                )

    @pytest.mark.parametrize("unbal_provider_large", [0, 1, 2], indirect=True)
    @patch("nndet.io.datamodule.mixins.select.load_pickle")
    def test_select_ratio(self, mock_load, unbal_provider_large):
        np.random.seed(0)  # seed selection

        mock_load.side_effect = unbal_provider_large.side_effect
        unbal_provider_large.cache = RandomSelectionMixin.build_cache(
            unbal_provider_large
        )
        classes_sampled = []
        for _ in range(self.select_iterations_full):
            cases, instance_ids = RandomSelectionMixin.select(unbal_provider_large)
            assert len(cases) == len(instance_ids)
            assert len(cases) == unbal_provider_large.batch_size
            _classes = [
                unbal_provider_large._data[c]["labels"][i - 1]
                for c, i in zip(cases, instance_ids)
                if i > -1
            ]
            classes_sampled.extend(_classes)
        _, cls_counts = np.unique(classes_sampled, return_counts=True)
        assert 0.88 <= cls_counts[0] / sum(cls_counts) <= 0.92
        assert 0.08 <= cls_counts[1] / sum(cls_counts) <= 0.12


class TestObjectBalancedSelection:
    select_iterations = 20
    select_iterations_full = 1000

    @patch("nndet.io.datamodule.mixins.select.load_pickle")
    def test_build_cache_no_instances(self, mock_load, no_instances_provider):
        mock_load.side_effect = no_instances_provider.side_effect
        cache = ObjectBalancedSelectionMixin.build_cache(no_instances_provider)
        mock_load.assert_called()

        assert cache["case"] == ["o1", "o2", "o3"]
        assert cache["bg"] == ["o1", "o2", "o3"]
        assert cache["fg"] == {}
        assert cache["sqrt_weights"] == {}

    @patch("nndet.io.datamodule.mixins.select.load_pickle")
    def test_build_cache_filled_instances(self, mock_load, filled_instances_provider):
        mock_load.side_effect = filled_instances_provider.side_effect
        cache = ObjectBalancedSelectionMixin.build_cache(filled_instances_provider)
        mock_load.assert_called()

        assert cache["case"] == ["o1", "o2", "o3", "o4", "o5", "o6"]
        assert cache["bg"] == ["o1", "o2", "o3"]
        assert cache["fg"] == {
            0: [("o4", 1), ("o6", 1), ("o6", 3)],
            1: [("o5", 1), ("o5", 2), ("o6", 2)],
        }
        assert cache["sqrt_weights"] == {0: 1 / 2, 1: 1 / 2}

    @patch("nndet.io.datamodule.mixins.select.load_pickle")
    def test_build_cache_unbal(self, mock_load, unbal_provider):
        mock_load.side_effect = unbal_provider.side_effect
        cache = ObjectBalancedSelectionMixin.build_cache(unbal_provider)
        mock_load.assert_called()

        assert cache["case"] == ["o1", "o2", "o3", "o4", "o5", "o6"]
        assert cache["bg"] == []
        assert cache["fg"] == {
            0: [("o4", 1), ("o6", 1), ("o6", 3)],
            1: [
                ("o1", 1),
                ("o1", 2),
                ("o2", 1),
                ("o2", 2),
                ("o3", 1),
                ("o3", 2),
                ("o5", 1),
                ("o5", 2),
                ("o6", 2),
            ],
        }
        # 0: 3 / 12; 1: 9 / 12
        assert len(cache["sqrt_weights"]) == 2
        assert math.isclose(
            cache["sqrt_weights"][0], math.sqrt(3) / (math.sqrt(3) + math.sqrt(9))
        )
        assert math.isclose(
            cache["sqrt_weights"][1], math.sqrt(9) / (math.sqrt(3) + math.sqrt(9))
        )

    @pytest.mark.parametrize("force_bg_case", [True, False])
    @pytest.mark.parametrize(
        "selection_mode", [SelectionMode("uniform"), SelectionMode("sqrt")]
    )
    @patch("nndet.io.datamodule.mixins.select.load_pickle")
    def test_select(
        self,
        mock_load,
        filled_instances_provider,
        force_bg_case: bool,
        selection_mode: SelectionMode,
    ):
        mock_load.side_effect = filled_instances_provider.side_effect
        filled_instances_provider.cache = ObjectBalancedSelectionMixin.build_cache(
            filled_instances_provider
        )
        filled_instances_provider.force_bg_case = force_bg_case
        filled_instances_provider.selection_mode = selection_mode

        for _ in range(self.select_iterations):
            cases, instance_ids = ObjectBalancedSelectionMixin.select(
                filled_instances_provider
            )
            assert (np.array(instance_ids) > 0).sum() == 2
            print(cases)
            if force_bg_case:
                assert all(
                    [
                        c in ["o1", "o2", "o3"]
                        for i, c in enumerate(cases)
                        if instance_ids[i] < 0
                    ]
                )

    @pytest.mark.parametrize("unbal_provider_large", [0, 1, 2], indirect=True)
    @patch("nndet.io.datamodule.mixins.select.load_pickle")
    def test_select_ratio(self, mock_load, unbal_provider_large):
        np.random.seed(0)  # seed selection

        mock_load.side_effect = unbal_provider_large.side_effect
        unbal_provider_large.cache = ObjectBalancedSelectionMixin.build_cache(
            unbal_provider_large
        )
        classes_sampled = []
        for _ in range(self.select_iterations_full):
            cases, instance_ids = ObjectBalancedSelectionMixin.select(
                unbal_provider_large
            )
            assert len(cases) == len(instance_ids)
            assert len(cases) == unbal_provider_large.batch_size
            _classes = [
                unbal_provider_large._data[c]["labels"][i - 1]
                for c, i in zip(cases, instance_ids)
                if i > -1
            ]
            classes_sampled.extend(_classes)
        _, cls_counts = np.unique(classes_sampled, return_counts=True)
        assert 0.48 <= cls_counts[0] / sum(cls_counts) <= 0.52
        assert 0.48 <= cls_counts[1] / sum(cls_counts) <= 0.52


class TestPatientBalancedSelection:
    select_iterations = 20
    select_iterations_full = 1000

    @patch("nndet.io.datamodule.mixins.select.load_pickle")
    def test_build_cache_no_instances(self, mock_load, no_instances_provider):
        mock_load.side_effect = no_instances_provider.side_effect
        cache = PatientBalancedSelectionMixin.build_cache(no_instances_provider)
        mock_load.assert_called()

        assert cache["case"] == ["o1", "o2", "o3"]
        assert cache["bg"] == ["o1", "o2", "o3"]
        assert cache["fg"] == {}
        assert cache["sqrt_weights"] == {}

    @patch("nndet.io.datamodule.mixins.select.load_pickle")
    def test_build_cache_filled_instances(self, mock_load, filled_instances_provider):
        mock_load.side_effect = filled_instances_provider.side_effect
        cache = PatientBalancedSelectionMixin.build_cache(filled_instances_provider)
        mock_load.assert_called()

        assert cache["case"] == ["o1", "o2", "o3", "o4", "o5", "o6"]
        assert cache["bg"] == ["o1", "o2", "o3"]
        assert cache["fg"] == {
            0: {"o4": [1], "o6": [1, 3]},
            1: {"o5": [1, 2], "o6": [2]},
        }
        assert cache["sqrt_weights"] == {0: 1 / 2, 1: 1 / 2}

    @patch("nndet.io.datamodule.mixins.select.load_pickle")
    def test_build_cache_unbal(self, mock_load, unbal_provider):
        mock_load.side_effect = unbal_provider.side_effect
        cache = PatientBalancedSelectionMixin.build_cache(unbal_provider)
        mock_load.assert_called()

        assert cache["case"] == ["o1", "o2", "o3", "o4", "o5", "o6"]
        assert cache["bg"] == []
        assert cache["fg"] == {
            0: {"o4": [1], "o6": [1, 3]},
            1: {"o1": [1, 2], "o2": [1, 2], "o3": [1, 2], "o5": [1, 2], "o6": [2]},
        }
        # 0: 2 / 7; 1: 5/ 7
        assert len(cache["sqrt_weights"]) == 2
        assert math.isclose(
            cache["sqrt_weights"][0], math.sqrt(2) / (math.sqrt(2) + math.sqrt(5))
        )
        assert math.isclose(
            cache["sqrt_weights"][1], math.sqrt(5) / (math.sqrt(2) + math.sqrt(5))
        )

    @pytest.mark.parametrize("force_bg_case", [True, False])
    @pytest.mark.parametrize(
        "selection_mode", [SelectionMode("uniform"), SelectionMode("sqrt")]
    )
    @patch("nndet.io.datamodule.mixins.select.load_pickle")
    def test_select(
        self,
        mock_load,
        filled_instances_provider,
        force_bg_case: bool,
        selection_mode: SelectionMode,
    ):
        mock_load.side_effect = filled_instances_provider.side_effect
        filled_instances_provider.cache = PatientBalancedSelectionMixin.build_cache(
            filled_instances_provider
        )
        filled_instances_provider.force_bg_case = force_bg_case
        filled_instances_provider.selection_mode = selection_mode

        for _ in range(self.select_iterations):
            cases, instance_ids = PatientBalancedSelectionMixin.select(
                filled_instances_provider
            )
            assert (np.array(instance_ids) > 0).sum() == 2
            print(cases)
            if force_bg_case:
                assert all(
                    [
                        c in ["o1", "o2", "o3"]
                        for i, c in enumerate(cases)
                        if instance_ids[i] < 0
                    ]
                )

    @pytest.mark.parametrize("unbal_provider_large", [0, 1, 2], indirect=True)
    @patch("nndet.io.datamodule.mixins.select.load_pickle")
    def test_select_ratio(self, mock_load, unbal_provider_large):
        np.random.seed(0)  # seed selection

        mock_load.side_effect = unbal_provider_large.side_effect
        unbal_provider_large.cache = PatientBalancedSelectionMixin.build_cache(
            unbal_provider_large
        )
        classes_sampled = []
        for _ in range(self.select_iterations_full):
            cases, instance_ids = PatientBalancedSelectionMixin.select(
                unbal_provider_large
            )
            assert len(cases) == len(instance_ids)
            assert len(cases) == unbal_provider_large.batch_size
            _classes = [
                unbal_provider_large._data[c]["labels"][i - 1]
                for c, i in zip(cases, instance_ids)
                if i > -1
            ]
            classes_sampled.extend(_classes)
        _, cls_counts = np.unique(classes_sampled, return_counts=True)
        assert 0.48 <= cls_counts[0] / sum(cls_counts) <= 0.52
        assert 0.48 <= cls_counts[1] / sum(cls_counts) <= 0.52
