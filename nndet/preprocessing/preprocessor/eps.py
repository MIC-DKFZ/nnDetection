# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence, Tuple

import numpy as np
from loguru import logger

from nndet.preprocessing.preprocessor.generic import GenericPreprocessor
from nndet.preprocessing.resampling import resample_patient


class EpsPreprocessor(GenericPreprocessor):
    DATA_ID = "EpsGeneric"

    def __init__(
        self,
        norm_scheme_per_modality: Dict[int, str],
        use_mask_for_norm: Dict[int, bool],
        transpose_forward: Sequence[int],
        intensity_properties: Dict[int, Dict] = None,
        resample_anisotropy_threshold: float = 3.0,
        resample_eps: float = 0.05,
    ):
        """
        Allows to skip resampling if relative difference (wrt. to target
        spacing) of target and original spacing is below a predefined epsilon.

        Args:
            norm_scheme_per_modality: integer index represents modality and string is
                either `CT`, `CT2`, 'BValRaw'. Other modalities are treated the with zeo mean and unit std.
            use_mask_for_norm: only foreground values should be used for normalization
                (defined for each modality)
            transpose_forward: transpose input data
            intensity_properties: Intensity properties of foreground over the dataset.
                Evaluated statistics: `median`; `mean`; `std`; `min`; `max`;
                `percentile_99_5`; `percentile_00_5`
                `local_props`: contains a dict (with case ids) where statistics
                where computed per case
            resample_eps: threshold of relative difference (wrt. to target
                spacing) when resampling should be performed.

        Overwrites:
            :self:`data_id`: unique identifier of GenericPreprocessor
        """
        super().__init__(
            norm_scheme_per_modality=norm_scheme_per_modality,
            use_mask_for_norm=use_mask_for_norm,
            transpose_forward=transpose_forward,
            intensity_properties=intensity_properties,
            resample_anisotropy_threshold=resample_anisotropy_threshold,
        )
        self.resample_eps = resample_eps

    def resample(
        self,
        data: np.ndarray,
        seg: np.ndarray,
        original_spacing: Sequence[float],
        target_spacing: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Resample data and segmentation to new spacing

        Args:
            data: input data
            seg: input segmentation
            original_spacing: original spacing
            target_spacing: target spacing

        Returns:
            np.ndarray: resampled data
            np.ndarray: resampled segmentation
            dict: properties after resampling
                `spacing`: spacing after resampling
                `shape (resampled)`: shape after resampling
        """
        original_spacing = np.array(original_spacing)
        target_spacing = np.array(target_spacing)
        data[np.isnan(data)] = 0

        rel_spacing = np.abs(target_spacing - original_spacing) / target_spacing
        if (rel_spacing <= self.resample_eps).all():
            logger.info("Spacing difference below eps, skipping reampling.")
            after = {
                "spacing": original_spacing,
                "shape (resampled)": data.shape,
            }
        else:
            data, seg = resample_patient(
                data,
                seg,
                original_spacing,
                target_spacing,
                order_data=3,
                order_seg=0,
                force_separate_z=False,
                order_z_data=9999,
                order_z_seg=9999,
                separate_z_anisotropy_threshold=self.resample_anisotropy_threshold,
            )

            after = {
                "spacing": target_spacing,
                "shape (resampled)": data.shape,
            }
        return data, seg, after
