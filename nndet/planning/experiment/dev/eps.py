# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import numpy as np

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

    def determine_forward_backward_permutation(self, mode: str):
        """
        Try to determine position of transverse plane
        Result is saved into :param:`transpose_forward` and
        :param:`transpose_backward`.
        """
        # spacings = self.data_properties['all_spacings']
        # sizes = self.data_properties['all_sizes']

        target_spacing = self.determine_target_spacing(mode=mode)
        # new_sizes = [np.array(i) / target_spacing * np.array(j) for i, j in zip(spacings, sizes)]

        dims = len(target_spacing)

        assert dims == 3, "Found non 3D data"
        diffs = [
            [
                target_spacing[0] - target_spacing[1],
                target_spacing[0] - target_spacing[2],
            ],
            [
                target_spacing[1] - target_spacing[0],
                target_spacing[1] - target_spacing[2],
            ],
            [
                target_spacing[2] - target_spacing[0],
                target_spacing[2] - target_spacing[1],
            ],
        ]
        diffs = np.abs(diffs).sum(axis=1)  # [3] each entry contains the diff to all other axes

        if np.allclose(diffs, np.zeros_like(diffs)):
            # all axis are the same
            self.transpose_forward = [0, 1, 2]
        else:
            max_diff_axis = np.argmax(diffs)

            remaining_axes = [i for i in list(range(dims)) if i != max_diff_axis]
            # self.transpose_forward = remaining_axes + [max_spacing_axis] # y, x, z
            self.transpose_forward = [max_diff_axis] + remaining_axes  # z, y, x

        self.transpose_backward = [np.argwhere(np.array(self.transpose_forward) == i)[0][0] for i in range(dims)]
