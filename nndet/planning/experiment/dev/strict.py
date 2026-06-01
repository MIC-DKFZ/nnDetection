# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

from nndet.planning.experiment import PLANNER_REGISTRY
from nndet.planning.experiment.v001 import D3V001
from nndet.preprocessing.preprocessor.strict import (
    StrictFgPreprocessor,
    StrictFgPreprocessorDynDtype,
    StrictPreprocessor,
    StrictPreprocessorDynDtype,
)


@PLANNER_REGISTRY.register
class V001StrictNorm(D3V001):
    @staticmethod
    def create_preprocessor(plan: Dict) -> StrictPreprocessor:
        """
        Create Preprocessor
        """
        preprocessor = StrictPreprocessor(
            norm_scheme_per_modality=plan["normalization_schemes"],
            use_mask_for_norm=plan["use_mask_for_norm"],
            transpose_forward=plan["transpose_forward"],
            intensity_properties=plan["dataset_properties"]["intensity_properties"],
            resample_anisotropy_threshold=plan["resample_anisotropy_threshold"],
        )
        return preprocessor


@PLANNER_REGISTRY.register
class V001StrictNormDynDtype(D3V001):
    @staticmethod
    def create_preprocessor(plan: Dict) -> StrictPreprocessorDynDtype:
        """
        Create Preprocessor
        """
        preprocessor = StrictPreprocessorDynDtype(
            norm_scheme_per_modality=plan["normalization_schemes"],
            use_mask_for_norm=plan["use_mask_for_norm"],
            transpose_forward=plan["transpose_forward"],
            intensity_properties=plan["dataset_properties"]["intensity_properties"],
            resample_anisotropy_threshold=plan["resample_anisotropy_threshold"],
        )
        return preprocessor


@PLANNER_REGISTRY.register
class V001StrictFgNorm(D3V001):
    @staticmethod
    def create_preprocessor(plan: Dict) -> StrictFgPreprocessor:
        """
        Create Preprocessor
        """
        preprocessor = StrictFgPreprocessor(
            norm_scheme_per_modality=plan["normalization_schemes"],
            use_mask_for_norm=plan["use_mask_for_norm"],
            transpose_forward=plan["transpose_forward"],
            intensity_properties=plan["dataset_properties"]["intensity_properties"],
            resample_anisotropy_threshold=plan["resample_anisotropy_threshold"],
        )
        return preprocessor


@PLANNER_REGISTRY.register
class V001StrictFgNormDynDtype(D3V001):
    @staticmethod
    def create_preprocessor(plan: Dict) -> StrictFgPreprocessorDynDtype:
        """
        Create Preprocessor
        """
        preprocessor = StrictFgPreprocessorDynDtype(
            norm_scheme_per_modality=plan["normalization_schemes"],
            use_mask_for_norm=plan["use_mask_for_norm"],
            transpose_forward=plan["transpose_forward"],
            intensity_properties=plan["dataset_properties"]["intensity_properties"],
            resample_anisotropy_threshold=plan["resample_anisotropy_threshold"],
        )
        return preprocessor
