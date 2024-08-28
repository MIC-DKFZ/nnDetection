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
        generator_patch_size: ND_TUPLE_INT,
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
            generator_patch_size: patch size before SpatialAugmentation
            magnitude: magnitude of offset
            p_per_sample: probability to apply transformation per sample
            fill_data: values filled for data
            fill_seg: values filled for segmentation
        """
        super().__init__()
        self.patch_size = np.array(patch_size)
        self.generator_patch_size = np.array(generator_patch_size)
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

                data_sample, seg_sample = self._random_crop(
                    data_sample,
                    seg_sample,
                    content_shape=content_shape,
                )
                data[self.data_key][batch_idx] = data_sample
                if self.seg_key is not None:
                    data[self.seg_key][batch_idx] = seg_sample
        return data

    def _random_crop(
        self,
        data: np.ndarray,
        seg: Optional[np.ndarray],
        content_shape: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Perform random shifting of data within bounds

        Args:
            data: data of chape, [c, x, y, z]
            seg: optionally provide segmentation of shape [c, x, y, z]

        Returns:
            np.ndarray: cropped and padded data
            np.ndarray: cropped and padded segmentation
        """
        dim = len(self.patch_size)
        content_difference = np.maximum(0, self.patch_size - content_shape) // 2

        for d in range(dim):
            # if == 0 content fills the entire patch -> no shifting
            if content_difference[d] > 0:
                # determine offset
                max_offset = self.magnitude * content_difference[d]
                offset = np.random.randint(-max_offset, max_offset)
                print(offset)

                # determine slices
                pg_center = self.generator_patch_size[d] / 2
                ps_extend = self.patch_size[d] / 2
                original_slices = slice(math.floor(pg_center - ps_extend), math.ceil(pg_center + ps_extend))
                shifted_slices = slice(
                    math.floor(pg_center - ps_extend + offset), math.ceil(pg_center + ps_extend + offset)
                )
                if offset > 0:
                    remaining_slices = slice(0, original_slices.start)
                else:
                    remaining_slices = slice(original_slices.stop, self.generator_patch_size[d])

                # shift data
                data[d][shifted_slices] = data[d][original_slices]
                data[d][remaining_slices] = self.fill_data
                if seg is not None:
                    seg[d][shifted_slices] = seg[d][original_slices]
                    seg[d][remaining_slices] = self.fill_seg
        return data, seg
