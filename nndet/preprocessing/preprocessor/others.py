# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
from loguru import logger

from nndet.io.load import load_case_cropped, save_pickle
from nndet.preprocessing.preprocessor.generic import GenericPreprocessor
from nndet.preprocessing.resampling import resample_patient


class Preprocessor2D(GenericPreprocessor):
    """
    Do not resample along the z direction
    """

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

        # prevent resampling along the z direction
        target_spacing[0] = original_spacing[0]

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


class Preprocessor2DRGB01(GenericPreprocessor):
    def normalize(self, data: np.ndarray, seg: np.ndarray) -> np.ndarray:
        """
        Normalize data by dividing by 255 to normalize to 0,1 range

        Args:
            data: input data
            seg: input data

        Returns:
            np.ndarray: normalized data
        """
        return data / 255


class PreprocessorNoResampling(GenericPreprocessor):
    def resample(
        self,
        data: np.ndarray,
        seg: np.ndarray,
        original_spacing: Sequence[float],
        target_spacing: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Do not resample

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
        data[np.isnan(data)] = 0

        after = {
            "spacing": original_spacing,
            "shape (resampled)": data.shape,
        }
        return data, seg, after


class PreprocessorRibFrac(PreprocessorNoResampling):
    """
    No resampling + bone window
    """

    def normalize_ct(
        self,
        data: np.ndarray,
        seg: np.ndarray,
        modality: int,
        use_nonzero_mask: Dict[int, bool],
    ) -> np.ndarray:
        """
        clip to lb and ub from train data foreground and use foreground mn and sd from training data
        (This uses the foreground mean and std!)
        Args:
            data: data to normalize [C, dims]
            seg: segmentation [C, dims]
            modality: current modality
            use_nonzero_mask: use non zero region for normalization and set all values
                outside to zero [C]

        Returns:
            np.ndarray: normalized data (only modality channel was changes)
        """
        lower_bound = 450.0 - (1100.0 / 2.0)
        upper_bound = 450.0 + (1100.0 / 2.0)
        data[modality] = np.clip(data[modality], lower_bound, upper_bound)
        data[modality] = data[modality] - data[modality].min()  # (0, X)
        data[modality] = data[modality] / data[modality].max()  # (0, 1)
        data[modality] = (data[modality] - 0.5) * 2  # (-1, 1)
        return data


class PreprocessorFP16I16(GenericPreprocessor):
    def run_process(
        self,
        target_spacing: Sequence[float],
        case_id: str,
        output_dir_stage: Path,
        cropped_data_dir: Path,
    ) -> None:
        """
        Process a single case
        Result is saved into :param:`output_dir_stage`

        Args:
            target_spacing: target spacing for processed case
            case_id: case identifier
            output_dir_stage: path to output directory
            cropped_data_dir: path to source directory
        """
        data, seg, properties = load_case_cropped(cropped_data_dir, case_id)
        seg = seg[None]

        data, seg, properties = self.apply_process(
            data, target_spacing, properties, seg
        )
        properties["use_nonzero_mask_for_norm"] = self.use_mask_for_norm

        data = data.astype(np.float16)  # use float16 instead of float32
        seg = seg.astype(np.int16)  # use int16 instead of int32

        candidates = self.compute_candidates(
            data=data,
            seg=seg,
            properties=properties,
        )

        logger.info(f"Saving: {case_id} into {output_dir_stage}.")
        np.savez_compressed(
            str(output_dir_stage / f"{case_id}.npz"),
            data=data,
            seg=seg,
        )

        save_pickle(candidates, output_dir_stage / f"{case_id}_boxes.pkl")
        save_pickle(properties, output_dir_stage / f"{case_id}.pkl")
