import math
from abc import abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np
from loguru import logger

from nndet.io.load import load_pickle
from nndet.utils.enums import SelectionMode
from nndet.utils.info import maybe_verbose_iterable


class SelectionMixin:
    _data: Dict[str, Dict]  # save paths to data
    batch_size: int  # batch size to sample
    oversample_foreground_percent: float  # balance foreground and background
    cache: Dict[str, List]  # output return from `build_cache`

    @abstractmethod
    def build_cache(self) -> Dict[str, List]:
        """
        Parse information which will be used during the selection process

        Returns:
            Dict[str, List]: cache for sampling
        """
        raise NotImplementedError

    @abstractmethod
    def select(self) -> Tuple[List, List]:
        """
        Selects cases and instances. If instance id is `-1` a random background
        patch will be sampled.

        - Foreground sampling: sample uniformly from all the foreground classes
            and enforce the respective class while patch sampling.
        - Background sampling: We jsut sample a random case

        Returns:
            List: case identifiers
            List: instance ids. `id > 0` represents the foreground isntance
                to sample while `id = -1` indicates background patches
        """
        raise NotImplementedError


class RandomSelectionMixin(SelectionMixin):
    force_bg_case: bool

    def build_cache(self) -> Dict[str, List]:
        """
        Build up cache for sampling

        Returns:
            Dict[str, List]: cache for sampling

                ``"case"``: list with all case identifiers

                ``"bg"``: list with case ids which do not contain any object

                ``"instances"``: list with tuple of (case_id, instance_id)
        """
        instance_cache = []
        bg_cache = []

        logger.info("Building Sampling Cache for Dataloder")
        for case_id, item in maybe_verbose_iterable(
            self._data.items(), desc="Sampling Cache"
        ):
            instances = load_pickle(item["boxes_file"])["instances"]
            if instances:
                for instance_id in instances:
                    instance_cache.append((case_id, instance_id))
            else:
                bg_cache.append(case_id)
        return {
            "case": list(self._data.keys()),
            "bg": bg_cache,
            "instances": instance_cache,
        }

    def select(self) -> Tuple[List, List]:
        """
        Selects cases and instances. If instance id is `-1` a random background
        patch will be sampled.

        - Foreground sampling: sample uniformly from all the foreground classes
            and enforce the respective class while patch sampling.
        - Background sampling: We jsut sample a random case

        Returns:
            List: case identifiers
            List: instance ids. `id > 0` represents the foreground isntance
                to sample while `id = -1` indicates background patches
        """
        selected_cases = []
        selected_instances = []

        for idx in range(self.batch_size):
            if idx < round(self.batch_size * (1 - self.oversample_foreground_percent)):
                # sample bg
                if self.force_bg_case:  # force background from case without objects
                    selected_cases.append(np.random.choice(self.cache["bg"]))
                else:  # force background from random case
                    selected_cases.append(np.random.choice(self.cache["case"]))
                selected_instances.append(-1)
            else:
                # sample fg / select an instance
                idx = np.random.choice(range(len(self.cache["instances"])))
                _case, _instance_id = self.cache["instances"][idx]
                selected_cases.append(_case)
                selected_instances.append(int(_instance_id))
        return selected_cases, selected_instances


class ObjectBalancedSelectionMixin(SelectionMixin):
    force_bg_case: bool
    _selection_mode: SelectionMode

    @property
    def selection_mode(self) -> SelectionMode:
        return self._selection_mode

    @selection_mode.setter
    def selection_mode(self, key: Union[str, SelectionMode]):
        self._selection_mode = SelectionMode(key)

    def build_cache(self) -> Dict[str, Union[Dict, List]]:
        """
        Build up cache for sampling

        Returns:
            Dict[str, Union[Dict, List]]: object and case cache

                ``"fg"`` Dict[int, Tuple[str, int]]: caches for foreground objects
                    for each class. The tuple saved the case id and instance
                    id.

                ``"bg"`` List[str]: all case ids without objects

                ``"case"`` List[str]: all case ids

                ``"sqrt_weights"`` Dict[int, float]: square root of number of
                    occurances normalized to sum to one
        """
        fg_cache = defaultdict(list)
        bg_cache = []

        logger.info("Building Sampling Cache for Dataloder")
        for case_id, item in maybe_verbose_iterable(
            self._data.items(), desc="Sampling Cache"
        ):
            candidates = load_pickle(item["boxes_file"])
            if candidates["instances"]:
                for instance_id, instance_class in zip(
                    candidates["instances"], candidates["labels"]
                ):
                    fg_cache[int(instance_class)].append((case_id, instance_id))
            else:
                bg_cache.append(case_id)

        sqrt_weights = {k: math.sqrt(len(i)) for k, i in fg_cache.items()}
        sum_all = sum(sqrt_weights.values())
        sqrt_weights = {k: i / sum_all for k, i in sqrt_weights.items()}

        assert list(sqrt_weights.keys()) == list(fg_cache.keys())
        return {
            "fg": fg_cache,
            "bg": bg_cache,
            "case": list(self._data.keys()),
            "sqrt_weights": sqrt_weights,
        }

    def select(self) -> Tuple[List, List]:
        """
        Selects cases and instances. If instance id is `-1` a random background
        patch will be sampled. In this balanced version, the classes
        are also balanced on an object level while this is not the case
        in the default loader (which leads to unbalanced results for
        data sets where foreground and background patches are unbalanced on
        a patient level and/or object level).

        - Foreground sampling: sample from all the foreground classes and
            than from objects of that class (as defined by mode).
        - Background sampling: We jsut sample a random case. If
            `force_bg_case=True` the random patch is sampled from a
            case without instances.

        Returns:
            List: case identifiers
            List: instance ids. `id > 0` represents the foreground instance
                to sample while `id = -1` indicates background patches
        """
        # randomly select foreground classes
        if self.selection_mode == SelectionMode.SQRT:
            selected_classes = np.random.choice(
                list(self.cache["sqrt_weights"].keys()),
                self.batch_size,
                replace=True,
                p=list(self.cache["sqrt_weights"].values()),
            )
        elif self.selection_mode == SelectionMode.UNIFORM:
            selected_classes = np.random.choice(
                list(self.cache["fg"].keys()),
                self.batch_size,
                replace=True,
            )
        else:
            raise RuntimeError(f"Unknown selection mode {self.selection_mode}.")

        selected_cases = []
        selected_instances = []
        for idx in range(len(selected_classes)):
            if idx < round(self.batch_size * (1 - self.oversample_foreground_percent)):
                # sample bg
                if self.force_bg_case:  # force background from case without objects
                    selected_cases.append(np.random.choice(self.cache["bg"]))
                else:  # force background from random case
                    selected_cases.append(np.random.choice(self.cache["case"]))
                selected_instances.append(-1)
            else:
                # sample fg / select an instance
                _class_cases: List[Tuple[str, int]] = self.cache["fg"][
                    selected_classes[idx]
                ]
                _i = np.random.choice(range(len(_class_cases)))
                _case, _instance_id = _class_cases[_i]
                selected_cases.append(_case)
                selected_instances.append(int(_instance_id))
        return selected_cases, selected_instances


class PatientBalancedSelectionMixin(SelectionMixin):
    force_bg_case: bool
    _selection_mode: SelectionMode

    @property
    def selection_mode(self) -> SelectionMode:
        return self._selection_mode

    @selection_mode.setter
    def selection_mode(self, key: Union[str, SelectionMode]):
        self._selection_mode = SelectionMode(key)

    def build_cache(self) -> Dict[str, Union[Dict, List]]:
        """
        Build up cache for sampling

        Returns:
            Dict[str, Union[Dict, List]]: object and case cache

                ``"fg"`` Dict[int, Dict[str, List[int]]]: caches for foreground
                    objects for each class. The outer dictionary contains the
                    object classes while the inner dict contains the case
                    ids. The list for each case id contains the instance
                    ids for the objects of the respective class (as defined
                    by the outer dict).

                ``"bg"`` List[str]: all case ids without objects

                ``"case"`` List[str]: all case ids

                ``"sqrt_weights"`` Dict[int, float]: square root of number of
                    occurances normalized to sum to one
        """
        fg_cache: Dict[Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        bg_cache: List[str] = []

        logger.info("Building Sampling Cache for Dataloder")
        for case_id, item in maybe_verbose_iterable(
            self._data.items(), desc="Sampling Cache"
        ):
            candidates = load_pickle(item["boxes_file"])
            if candidates["instances"]:
                for instance_id, instance_class in zip(
                    candidates["instances"], candidates["labels"]
                ):
                    fg_cache[int(instance_class)][case_id].append(instance_id)
            else:
                bg_cache.append(case_id)

        sqrt_weights = {k: math.sqrt(len(i)) for k, i in fg_cache.items()}
        sum_all = sum(sqrt_weights.values())
        sqrt_weights = {k: i / sum_all for k, i in sqrt_weights.items()}

        assert list(sqrt_weights.keys()) == list(fg_cache.keys())
        return {
            "fg": fg_cache,
            "bg": bg_cache,
            "case": list(self._data.keys()),
            "sqrt_weights": sqrt_weights,
        }

    def select(self) -> Tuple[List, List]:
        """
        Selects cases and instances. If instance id is `-1` a random background
        patch will be sampled. In this balanced version, the classes
        are also balanced on an patient level while this is not the case
        in the default loader (which leads to unbalanced results for
        data sets where foreground and background patches are unbalanced on
        a patient level and/or object level).

        - Foreground sampling: sample from all the foreground classes
            and than from cases which contains that class (as defined by mode).
            Objects of the selected class and case are sampled uniformly.
        - Background sampling: We jsut sample a random case. If
            `force_bg_case=True` the random patch is sampled from a
            case without instances.

        Returns:
            List: case identifiers
            List: instance ids. `id > 0` represents the foreground instance
                to sample while `id = -1` indicates background patches
        """
        # randomly select foreground classes
        if self.selection_mode == SelectionMode.SQRT:
            selected_classes = np.random.choice(
                list(self.cache["sqrt_weights"].keys()),
                self.batch_size,
                replace=True,
                p=list(self.cache["sqrt_weights"].values()),
            )
        elif self.selection_mode == SelectionMode.UNIFORM:
            selected_classes = np.random.choice(
                list(self.cache["fg"].keys()),
                self.batch_size,
                replace=True,
            )
        else:
            raise RuntimeError(f"Unknown selection mode {self.selection_mode}.")

        selected_cases = []
        selected_instances = []
        for idx in range(len(selected_classes)):
            if idx < round(self.batch_size * (1 - self.oversample_foreground_percent)):
                # sample bg
                if self.force_bg_case:  # force background from case without objects
                    selected_cases.append(np.random.choice(self.cache["bg"]))
                else:  # force background from random case
                    selected_cases.append(np.random.choice(self.cache["case"]))
                selected_instances.append(-1)
            else:
                # sample fg / select an instance
                _class_cases: Dict[str, List[int]] = self.cache["fg"][
                    selected_classes[idx]
                ]
                _case = np.random.choice(list(_class_cases.keys()))
                _instance_id = np.random.choice(_class_cases[_case])
                selected_cases.append(_case)
                selected_instances.append(int(_instance_id))
        return selected_cases, selected_instances
