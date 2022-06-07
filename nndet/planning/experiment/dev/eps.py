# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

from nndet.planning.experiment import PLANNER_REGISTRY
from nndet.planning.experiment.v001 import D3V001
from nndet.preprocessing.preprocessor.eps import EpsPreprocessor


@PLANNER_REGISTRY.register
class V001Eps05(D3V001):
    @staticmethod
    def create_preprocessor(plan: Dict) -> EpsPreprocessor:
        """
        Create Preprocessor
        """
        preprocessor = EpsPreprocessor(
            norm_scheme_per_modality=plan["normalization_schemes"],
            use_mask_for_norm=plan["use_mask_for_norm"],
            transpose_forward=plan["transpose_forward"],
            intensity_properties=plan["dataset_properties"]["intensity_properties"],
            resample_anisotropy_threshold=plan["resample_anisotropy_threshold"],
            resample_eps=0.05,
        )
        return preprocessor
