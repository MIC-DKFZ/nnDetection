# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from loguru import logger

from nndet.preprocessing.preprocessor.generic import (
    DynDTypePreprocessor,
    GenericPreprocessor,
)


class StrictNormMixin:
    def init_norm_schemes(self):
        norm_schemes = {
            "ct": self.normalize_ct,
            "ct2": self.normalize_ct2,
            "raw": self.no_norm,
        }

        rep = "Found normalisation schemes: "
        for idx, c in self.norm_scheme_per_modality.items():
            n = c.lower() if c.lower() in norm_schemes else "other"
            rep = rep + f"({idx}): {n}"
        logger.info(rep)
        logger.info(f"Use mask for norm: {self.use_mask_for_norm}")
        return norm_schemes

    def normalize(self, data: np.ndarray, seg: np.ndarray) -> np.ndarray:
        """
        Normalize data with correct scheme

        Args:
            data: input data
            seg: input data

        Returns:
            np.ndarray: normalized data
        """
        assert len(self.norm_scheme_per_modality) == len(
            data
        ), "norm_scheme_per_modality must have as many entries as data has modalities"
        assert len(self.use_mask_for_norm) == len(
            data
        ), "use_mask_for_norm must have as many entries as data has modalities"

        for c in range(len(data)):
            scheme = self.norm_scheme_per_modality[c].lower()

            if scheme in self.norm_schemes:
                scheme_fn = self.norm_schemes[scheme]
            else:
                scheme_fn = self.normalize_other
            data[c] = scheme_fn(data[c], seg, c, self.use_mask_for_norm[c])
        return data

    def normalize_ct(
        self,
        data: np.ndarray,
        seg: np.ndarray,
        modality: int,
        use_nonzero_mask: bool,
    ) -> np.ndarray:
        """
        clip to lb and ub from train data foreground and use foreground mn and sd from training data
        (This uses the foreground mean and std!)
        Args:
            data: data to normalize [dims]
            seg: segmentation [C, dims]
            modality: current modality
            use_nonzero_mask: use non zero region for normalization and set
                all values outside to zero [C]

        Returns:
            np.ndarray: normalized data (only modality channel was changes)
        """
        assert (
            self.intensity_properties is not None
        ), "ERROR: if there is a CT then we need intensity properties"
        mean_intensity = self.intensity_properties[modality]["mean"]
        std_intensity = self.intensity_properties[modality]["std"]
        lower_bound = self.intensity_properties[modality]["percentile_00_5"]
        upper_bound = self.intensity_properties[modality]["percentile_99_5"]

        data = np.clip(data, lower_bound, upper_bound)
        data = (data - mean_intensity) / std_intensity

        if use_nonzero_mask:
            data[seg[-1] < 0] = 0
        return data

    def normalize_ct2(
        self,
        data: np.ndarray,
        seg: np.ndarray,
        modality: int,
        use_nonzero_mask: bool,
    ) -> np.ndarray:
        """
        clip to lb and ub from train data foreground, use mn and sd
        from each case for normalization
        (This uses mean and std from whole case!)

        Args:
            data: data to normalize [C, dims]
            seg: segmentation [C, dims]
            modality: current modality
            use_nonzero_mask: use non zero region for normalization [C]

        Returns:
            np.ndarray: normalized data (only modality channel was changes)
        """
        assert (
            self.intensity_properties is not None
        ), "ERROR: if there is a CT then we need intensity properties"
        lower_bound = self.intensity_properties[modality]["percentile_00_5"]
        upper_bound = self.intensity_properties[modality]["percentile_99_5"]

        mask = (data > lower_bound) & (data < upper_bound)
        data = np.clip(data, lower_bound, upper_bound)

        mn = data[mask].mean()
        sd = data[mask].std()
        data = (data - mn) / sd

        if use_nonzero_mask:
            data[seg[-1] < 0] = 0
        return data

    def normalize_other(
        self,
        data: np.ndarray,
        seg: np.ndarray,
        modality: int,
        use_nonzero_mask: bool,
    ) -> np.ndarray:
        """
        Zero mean and unit std

        Args:
            data: data to normalize [C, dims]
            seg: segmentation [C, dims]
            modality: current modality
            use_nonzero_mask: use non zero region for normalization [C]

        Returns:
            np.ndarray: normalized data (only modality channel was changes)
        """
        data = (data - data.mean()) / (data.std() + 1e-8)

        if use_nonzero_mask:
            data[seg[-1] < 0] = 0
        return data

    def no_norm(
        self,
        data: np.ndarray,
        seg: np.ndarray,
        modality: int,
        use_nonzero_mask: bool,
    ) -> np.ndarray:
        """
        No normalization only masking

        Args:
            data: data to normalize [C, dims]
            seg: segmentation [C, dims]
            modality: current modality
            use_nonzero_mask: use non zero region for normalization [C]

        Returns:
            np.ndarray: masked data (only modality channel was changed)
        """
        if use_nonzero_mask:
            data[seg[-1] < 0] = 0
        return data


class StrictNormFgMixin(StrictNormMixin):
    def normalize_other(
        self,
        data: np.ndarray,
        seg: np.ndarray,
        modality: int,
        use_nonzero_mask: bool,
    ) -> np.ndarray:
        """
        Zero mean and unit std computed on FG mask

        Args:
            data: data to normalize [C, dims]
            seg: segmentation [C, dims]
            modality: current modality
            use_nonzero_mask: use non zero region for normalization [C]

        Returns:
            np.ndarray: normalized data (only modality channel was changes)
        """
        mask = data > 0
        mn = data[mask].mean()
        sd = data[mask].std()

        data = (data - mn) / (sd + 1e-8)

        if use_nonzero_mask:
            data[seg[-1] < 0] = 0
        return data


class StrictPreprocessor(
    StrictNormMixin,
    GenericPreprocessor,
):
    pass


class StrictPreprocessorDynDtype(
    StrictNormMixin,
    DynDTypePreprocessor,
):
    pass


class StrictFgPreprocessor(
    StrictNormFgMixin,
    GenericPreprocessor,
):
    pass


class StrictFgPreprocessorDynDtype(
    StrictNormFgMixin,
    DynDTypePreprocessor,
):
    pass
