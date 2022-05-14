from typing import List

import pytest

from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.utils.typing import ND_TUPLE_INT


class StaticStrideSeqBackbone(AbstractBackbone):
    def get_strides(self) -> List[ND_TUPLE_INT]:
        return [
            (16, 32, 64),
            (32, 47, 128),
        ]


class StaticStrideIntBackbone(AbstractBackbone):
    def get_strides(self) -> List[ND_TUPLE_INT]:
        return [
            (16, 32, 64),
            55,
        ]


CASES_SEQ = [
    ((64, 94, 512), True),
    ((64, 94, 513), False),
    ((64, 92, 512), False),
    ((33, 94, 512), False),
    ((30, 94, 512), False),
]
CASES_INT = [
    ((110, 220, 55), True),
    ((111, 220, 55), False),
    ((110, 230, 55), False),
    ((110, 220, 60), False),
    ((110, 220, 50), False),
]


@pytest.mark.parametrize("patch_size,expected_result", CASES_SEQ)
def test_check_patch_size_seq(patch_size, expected_result):
    backbone = StaticStrideSeqBackbone()
    compat = backbone.check_patch_size(patch_size)
    assert compat == expected_result


@pytest.mark.parametrize("patch_size,expected_result", CASES_INT)
def test_check_patch_size_int(patch_size, expected_result):
    backbone = StaticStrideIntBackbone()
    compat = backbone.check_patch_size(patch_size)
    assert compat == expected_result


def test_check_patch_size_error():
    backbone = StaticStrideSeqBackbone()
    with pytest.raises(ValueError):
        compat = backbone.check_patch_size((64, 94))
