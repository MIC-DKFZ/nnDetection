from abc import abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger

from nndet.io.load import load_pickle
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
    def build_cache(self) -> Dict[str, List]:
        """
        Build up cache for sampling

        Returns:
            Dict[str, List]: cache for sampling

                ``"case"``: list with all case identifiers

                ``"instances"``: list with tuple of (case_id, instance_id)
        """
        instance_cache = []

        logger.info("Building Sampling Cache for Dataloder")
        for case_id, item in maybe_verbose_iterable(
            self._data.items(), desc="Sampling Cache"
        ):
            instances = load_pickle(item["boxes_file"])["instances"]
            if instances:
                for instance_id in instances:
                    instance_cache.append((case_id, instance_id))
        return {"case": list(self._data.keys()), "instances": instance_cache}

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
                # sample bg / random case
                selected_cases.append(np.random.choice(self.cache["case"]))
                selected_instances.append(-1)
            else:
                # sample fg / select an instance
                idx = np.random.choice(range(len(self.cache["instances"])))
                _case, _instance_id = self.cache["instances"][idx]
                selected_cases.append(_case)
                selected_instances.append(int(_instance_id))
        return selected_cases, selected_instances


class BalancedSelectionMixin(SelectionMixin):
    def build_cache(self) -> Tuple[Dict[int, List[Tuple[str, int]]], List]:
        """
        Build up cache for sampling

        Returns:
            Dict[int, List[Tuple[str, int]]]: foreground cache which contains
                of list of tuple of case ids and instance ids for each class
            List: background cache (all samples which do not have any
                foreground)
        """
        fg_cache = defaultdict(list)

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
        return {"fg": fg_cache, "case": list(self._data.keys())}

    def select(self) -> Tuple[List, List]:
        """
        Selects cases and instances. If instance id is `-1` a random background
        patch will be sampled. In this balanced version, the classes
        are also balanced on a patient level while this is not the case
        in the default loader (which leads to unbalanced results for
        data sets where foreground and background patches are unbalanced on
        a patient level).

        - Foreground sampling: sample uniformly from all the foreground classes
            and enforce the respective class while patch sampling.
        - Background sampling: We jsut sample a random case

        Returns:
            List: case identifiers
            List: instance ids. `id > 0` represents the foreground instance
                to sample while `id = -1` indicates background patches
        """
        # randomly select foreground classes
        selected_classes = np.random.choice(
            list(self.cache["fg"].keys()), self.batch_size, replace=True
        )

        selected_cases = []
        selected_instances = []
        for idx in range(len(selected_classes)):
            if idx < round(self.batch_size * (1 - self.oversample_foreground_percent)):
                # sample bg / random case
                selected_cases.append(np.random.choice(self.cache["case"]))
                selected_instances.append(-1)
            else:
                # sample fg / select an instance
                _i = np.random.choice(
                    range(len(self.cache["fg"][selected_classes[idx]]))
                )
                _case, _instance_id = self.cache["fg"][selected_classes[idx]][_i]
                selected_cases.append(_case)
                selected_instances.append(int(_instance_id))
        return selected_cases, selected_instances


"""
# TODOs:

Pat Balanced Sampler
Obj Balanced Sampler

modes: square root vs uniform
Background: how to balance background images, e.g. force background from background image
"""
