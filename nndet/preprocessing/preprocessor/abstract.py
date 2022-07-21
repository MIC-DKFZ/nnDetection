# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Sequence, Tuple, TypeVar

import numpy as np


class AbstractPreprocessor(ABC):
    DATA_ID = "abstractdata"

    def __init__(self, **kwargs):
        """
        Interface for preprocessor
        """
        for key, item in kwargs.items():
            setattr(self, key, item)

    @abstractmethod
    def run(
        self,
        target_spacings: Sequence[Sequence[float]],
        identifiers: Sequence[str],
        cropped_data_dir: Path,
        preprocessed_output_dir: Path,
        num_processes: int,
        force_separate_z=None,
    ):
        """
        Run preprocessing

        Args:
            target_spacings: target spacing for each case
            identifiers: identifier strings used to name the directory
            cropped_data_dir: source directory
            preprocessed_output_dir: target directory
            num_processes: number of processes used for preprocessing
            force_separate_z: force independent resampling of z direction
        """
        raise NotImplementedError

    @abstractmethod
    def run_test(
        self,
        data_files,
        target_spacing,
        target_dir: PathLike,
    ) -> None:
        """
        Preprocess and save test data

        Args:
            data_files: path to data files
            target_spacing: spacing to resample
            target_dir: directory to save data to
        """
        raise NotImplementedError

    @abstractmethod
    def preprocess_test_case(
        self,
        data_files,
        target_spacing,
        seg_file=None,
        force_separate_z=None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Preprocess a test file

        Args:
            data_files: path to data files
            target_spacing: spacing to resample
            seg_file: optional segmentation file
            force_separate_z: separate resampling in z direction

        Returns:
            np.ndarray: preprocessed data [C, dims]
            np.ndarray: preprocessed segmentation [1, dims]
            dict: updated properties
        """
        raise NotImplementedError


PreprocessorType = TypeVar("PreprocessorType", bound=AbstractPreprocessor)
