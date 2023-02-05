from typing import Dict
from unittest.mock import call, patch

import numpy as np
import pytest

from nndet.io.datamodule.mixins.bgcrop import RandomBGCrop3D

SEEDS = [0, 1, 2, 3, 4, 5]


@pytest.fixture
def crop_kwargs() -> Dict:
    return {
        "case_data": np.zeros((2, 16, 32, 64)),
        # "case_seg": np.zeros((1, 16, 32, 64)),
        "properties": {},
        "case_id": "test_id",
    }


@pytest.fixture
def cropper() -> RandomBGCrop3D:
    a = RandomBGCrop3D()
    a.patch_size_generator = [12, 16, 32]
    a.patch_size_final = [6, 10, 20]
    a.need_to_pad = [6, 6, 12]
    return a


class TestRandomBGCrop:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_bg_offset_smoke(self, seed, cropper: RandomBGCrop3D, crop_kwargs: Dict):
        np.random.seed(seed)

        for i in range(100):  # perform 100 random crops
            crop = cropper.get_bg_crop(
                **crop_kwargs,
                candidates=None,
            )

            assert len(crop) == 3
            assert (crop[0].stop - crop[0].start) == cropper.patch_size_generator[0]
            assert (crop[1].stop - crop[1].start) == cropper.patch_size_generator[1]
            assert (crop[2].stop - crop[2].start) == cropper.patch_size_generator[2]

            assert crop[0].start >= -3
            assert crop[0].stop <= 19
            assert crop[1].start >= -3
            assert crop[1].stop <= 35
            assert crop[2].start >= -6
            assert crop[2].stop <= 70
