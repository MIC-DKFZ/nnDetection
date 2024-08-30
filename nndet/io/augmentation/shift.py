# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional, Tuple

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform

from nndet.utils.typing import ND_TUPLE_INT


class RandomShiftTransform(AbstractTransform):
    def __init__(
        self,
        data_key: str,
        patch_size: ND_TUPLE_INT,
        magnitude: float = 1.0,
        p_per_sample: float = 1.0,
        seg_key: Optional[str] = None,
        fill_data: float = 0,
        fill_seg: float = -1,
    ) -> None:
        """
        This transformation should be inserted in front of the SpatialTransform
        to perform an offset first. Requires "properties/size_after_resampling"
        keys in the batch to determine maximum offset. If the original
        data is not filling up the entire patch, it will shift
        it within the patch size.

        Args:
            keys: keys to crop
            patch_size: patch size after SpatialAugmentation
            magnitude: magnitude of offset
            p_per_sample: probability to apply transformation per sample
            fill_data: values filled for data
            fill_seg: values filled for segmentation
        """
        super().__init__()
        self.patch_size = np.array(patch_size)
        self.magnitude = magnitude
        self.p_per_sample = p_per_sample

        self.fill_data = fill_data
        self.fill_seg = fill_seg

        self.data_key = data_key
        self.seg_key = seg_key

    def __call__(self, **data) -> dict:
        batch_size = len(data[self.data_key])

        for batch_idx in range(batch_size):
            if np.random.random() < self.p_per_sample:
                data_sample = data[self.data_key][batch_idx]
                seg_sample = data[self.seg_key][batch_idx] if self.seg_key is not None else None
                content_shape = np.array(data["properties"][batch_idx]["size_after_resampling"])
                generator_patch_size = np.array(data_sample.shape[1:])

                data_sample, seg_sample = self._random_shift(
                    data=data_sample,
                    content_shape=content_shape,
                    generator_patch_size=generator_patch_size,
                    seg=seg_sample,
                )
                data[self.data_key][batch_idx] = data_sample
                if self.seg_key is not None:
                    data[self.seg_key][batch_idx] = seg_sample
        return data

    def _random_shift(
        self,
        data: np.ndarray,
        generator_patch_size: np.ndarray,
        content_shape: np.ndarray,
        seg: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Perform random shifting of data within bounds

        Args:
            data: data of chape, [c, x, y, z]
            generator_patch_size: patch size of the incoming data
            content_shape: shape of the original data actually containing
                content
            seg: optionally provide segmentation of shape [c, x, y, z]

        Returns:
            np.ndarray: cropped and padded data
            np.ndarray: cropped and padded segmentation
        """
        dim = len(self.patch_size)
        # estimate lower and upper bound of offset
        content_difference = np.maximum(0, self.patch_size - content_shape) / 2
        max_content_difference = np.floor(content_difference)

        for d in range(dim):
            # if == 0 content fills the entire patch -> no shifting
            if content_difference[d] > 0:
                # determine offset
                max_offset = self.magnitude * content_difference[d]
                offset = np.random.randint(-np.floor(max_offset), np.ceil(max_offset))
                if offset < 0:
                    offset = int(np.maximum(max_content_difference[d], offset))
                else:
                    offset = int(np.minimum(max_content_difference[d], offset))

                # determine slices
                pg_center = generator_patch_size[d] / 2
                ps_extend = self.patch_size[d] / 2

                lw = math.floor(pg_center - ps_extend)
                up = math.ceil(pg_center + ps_extend)
                original_slices = slice(lw, up)
                shifted_slices = slice(lw + offset, up + offset)
                if offset > 0:
                    remaining_slices = slice(0, original_slices.start)
                else:
                    remaining_slices = slice(original_slices.stop, generator_patch_size[d])

                # shift data
                data[d][shifted_slices] = data[d][original_slices]
                data[d][remaining_slices] = self.fill_data
                if seg is not None:
                    seg[d][shifted_slices] = seg[d][original_slices]
                    seg[d][remaining_slices] = self.fill_seg
        return data, seg
