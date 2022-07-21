# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Sequence, Tuple


def to_nd_tuple(x: Any, dim: int) -> Tuple[Any]:
    """
    Ensure (none-sequence!) input is in n-D tuple format

    Args:
        x: input to check
        dim: number of spatial dimensions

    Returns:
        Tuple: n-D tuple
    """
    if not isinstance(x, Sequence):
        return tuple([x] * dim)
    else:
        if len(x) != dim:
            raise ValueError(f"Expectd {x} to have {dim} entries for nd-tuple.")
        return tuple(x)
