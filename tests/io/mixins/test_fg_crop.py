from typing import Dict
from unittest.mock import Mock, call, patch

import numpy as np
import pytest

from nndet.io.datamodule.mixins.fgcrop import OffsetFGCrop3D, OffsetFGCrop3DV2
from nndet.io.patching import save_get_crop

SEEDS = [0, 1, 2, 3, 4, 5]


@pytest.fixture
def cropper() -> OffsetFGCrop3DV2:
    a = OffsetFGCrop3DV2()
    a.offset_prob = 1.0
    a.offset_magn = 1.0
    a.patch_size_generator = [12, 16, 32]
    a.patch_size_final = [6, 10, 20]
    a.need_to_pad = [6, 6, 12]
    a.max_size_pct = 1.0
    return a


@pytest.fixture
def crop_kwargs() -> Dict:
    return {
        "case_data": np.zeros((2, 16, 32, 64)),
        "case_seg": np.zeros((1, 16, 32, 64)),
        "properties": {},
        "case_id": "test_id",
    }


@pytest.fixture
def candidates() -> Dict:
    return {
        "instances": [1, 2, 3, 4],
        "boxes": np.array(
            [
                [
                    6,
                    14,
                    10,
                    18,
                    30,
                    34,
                ],  # center-ish
                [
                    -1,
                    0,
                    3,
                    8,
                    1,
                    16,
                ],  # lower bound
                [
                    12,
                    28,
                    16,
                    32,
                    58,
                    64,
                ],  # upper bound
                [
                    -1,
                    -1,
                    16,
                    32,
                    -1,
                    64,
                ],  # as big as whole scan
            ]
        ),
    }


class TestOffsetFGCrop3DV2:
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("instance_id", [1, 2, 3])
    def test_call_no_offset_smoke(
        self,
        seed: int,
        instance_id: int,
        cropper: OffsetFGCrop3DV2,
        crop_kwargs: Dict,
        candidates: Dict,
    ):
        np.random.seed(seed)
        mock = Mock(side_effect=cropper._inside_box)
        cropper.offset_prob = 0.0
        cropper._inside_box = mock

        crop = cropper.get_fg_crop(
            **crop_kwargs,
            instance_id=instance_id,
            candidates=candidates,
        )

        assert len(crop) == 3
        assert (crop[0].stop - crop[0].start) == cropper.patch_size_generator[0]
        assert (crop[1].stop - crop[1].start) == cropper.patch_size_generator[1]
        assert (crop[2].stop - crop[2].start) == cropper.patch_size_generator[2]

        assert crop[0].start <= candidates["boxes"][instance_id - 1, 0] <= crop[0].stop
        assert crop[0].start <= candidates["boxes"][instance_id - 1, 2] <= crop[0].stop
        assert crop[1].start <= candidates["boxes"][instance_id - 1, 1] <= crop[1].stop
        assert crop[1].start <= candidates["boxes"][instance_id - 1, 3] <= crop[1].stop
        assert crop[2].start <= candidates["boxes"][instance_id - 1, 4] <= crop[2].stop
        assert crop[2].start <= candidates["boxes"][instance_id - 1, 5] <= crop[2].stop

        mock.assert_called()
        box = candidates["boxes"][instance_id - 1]
        calls = [
            call(ps=6, psg=12, box_lower=box[0], box_upper=box[2]),
            call(ps=10, psg=16, box_lower=box[1], box_upper=box[3]),
            call(ps=20, psg=32, box_lower=box[4], box_upper=box[5]),
        ]
        mock.assert_has_calls(calls)

    @pytest.mark.parametrize("big_object", [True, False])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_call_big_object(
        self,
        seed: int,
        cropper: OffsetFGCrop3DV2,
        crop_kwargs: Dict,
        candidates: Dict,
        big_object: bool,
    ):
        np.random.seed(seed)
        mock = Mock(side_effect=cropper._inside_box)
        cropper._inside_box = mock

        if big_object:
            instance_id = 4
        else:
            instance_id = 1
            cropper.max_size_pct = 0.1

        crop = cropper.get_fg_crop(
            **crop_kwargs,
            instance_id=instance_id,
            candidates=candidates,
        )

        assert len(crop) == 3
        assert (crop[0].stop - crop[0].start) == cropper.patch_size_generator[0]
        assert (crop[1].stop - crop[1].start) == cropper.patch_size_generator[1]
        assert (crop[2].stop - crop[2].start) == cropper.patch_size_generator[2]

        mock.assert_called()
        box = candidates["boxes"][instance_id - 1]
        calls = [
            call(ps=6, psg=12, box_lower=box[0], box_upper=box[2]),
            call(ps=10, psg=16, box_lower=box[1], box_upper=box[3]),
            call(ps=20, psg=32, box_lower=box[4], box_upper=box[5]),
        ]
        mock.assert_has_calls(calls)

    def test_call_center_data(
        self,
        cropper: OffsetFGCrop3DV2,
        crop_kwargs: Dict,
        candidates: Dict,
    ):
        mock = Mock(side_effect=cropper._center_data)
        cropper._center_data = mock
        cropper.patch_size_final = [128, 256, 512]
        cropper.patch_size_generator = [256, 512, 1024]

        crop = cropper.get_fg_crop(
            **crop_kwargs,
            instance_id=1,
            candidates=candidates,
        )

        assert len(crop) == 3
        assert (crop[0].stop - crop[0].start) == cropper.patch_size_generator[0]
        assert (crop[1].stop - crop[1].start) == cropper.patch_size_generator[1]
        assert (crop[2].stop - crop[2].start) == cropper.patch_size_generator[2]

        mock.assert_called()
        calls = [
            call(ps=128, psg=256, spatial_size=16),
            call(ps=256, psg=512, spatial_size=32),
            call(ps=512, psg=1024, spatial_size=64),
        ]
        mock.assert_has_calls(calls)
        assert crop[0] == slice(-120, 136)
        assert crop[1] == slice(-240, 272)
        assert crop[2] == slice(-480, 544)

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("instance_id", [1, 2, 3])
    def test_call_offset_data(
        self,
        seed: int,
        instance_id: int,
        cropper: OffsetFGCrop3DV2,
        crop_kwargs: Dict,
        candidates: Dict,
    ):
        np.random.seed(seed)
        mock = Mock(side_effect=cropper._offset_box)
        cropper.offset_prob = 1.0
        cropper._offset_box = mock

        crop = cropper.get_fg_crop(
            **crop_kwargs,
            instance_id=instance_id,
            candidates=candidates,
        )

        assert len(crop) == 3
        assert (crop[0].stop - crop[0].start) == cropper.patch_size_generator[0]
        assert (crop[1].stop - crop[1].start) == cropper.patch_size_generator[1]
        assert (crop[2].stop - crop[2].start) == cropper.patch_size_generator[2]

        assert crop[0].start <= candidates["boxes"][instance_id - 1, 0] <= crop[0].stop
        assert crop[0].start <= candidates["boxes"][instance_id - 1, 2] <= crop[0].stop
        assert crop[1].start <= candidates["boxes"][instance_id - 1, 1] <= crop[1].stop
        assert crop[1].start <= candidates["boxes"][instance_id - 1, 3] <= crop[1].stop
        assert crop[2].start <= candidates["boxes"][instance_id - 1, 4] <= crop[2].stop
        assert crop[2].start <= candidates["boxes"][instance_id - 1, 5] <= crop[2].stop

        mock.assert_called()
        box = candidates["boxes"][instance_id - 1]
        calls = [
            call(
                ps=6,
                ntp=6,
                spatial_size=16,
                box_lower=box[0],
                box_upper=box[2],
            ),
            call(
                ps=10,
                ntp=6,
                spatial_size=32,
                box_lower=box[1],
                box_upper=box[3],
            ),
            call(
                ps=20,
                ntp=12,
                spatial_size=64,
                box_lower=box[4],
                box_upper=box[5],
            ),
        ]
        mock.assert_has_calls(calls)

    def test_offseet_box_magn0(self, cropper: OffsetFGCrop3DV2):
        cropper.offset_magn = 0.0
        idx = cropper._offset_box(ps=128, ntp=72, spatial_size=256, box_lower=95, box_upper=105)
        assert idx == 0

    @pytest.mark.parametrize("test_cls", [OffsetFGCrop3D, OffsetFGCrop3DV2])
    def test_issue_72(self, test_cls):
        cropper = test_cls()
        cropper.offset_prob = 1.0
        cropper.offset_magn = 1.0
        cropper.patch_size_generator = [352, 114, 114]
        cropper.patch_size_final = [256, 64, 64]
        cropper.need_to_pad = [6, 6, 12]
        cropper.max_size_pct = 1.0

        data = np.zeros((1, 332, 80, 294))
        candidates = {
            "boxes": np.array([[162, 6, 204, 48, 42, 126]]),
            "instances": [1],
            "labels": [0],
        }

        crop = cropper.get_fg_crop(
            case_data=data,
            case_seg=None,
            properties={},
            case_id="test0",
            candidates=candidates,
            instance_id=1,
        )

        patch = save_get_crop(
            data,
            crop=crop,
            mode="constant",
        )[0]
        assert tuple(patch.shape) == (1, 352, 114, 114)
