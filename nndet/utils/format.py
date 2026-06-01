# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Sequence, Tuple

import numpy as np


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


def make_plan_json_compatible(plan: dict) -> dict:
    """
    Make the plan json compatible: pop dataset properties, convert ndarray
    into list and cast some items to int, bool

    Args:
        plan: plan to convert

    Returns:
        dict: json compatible plan
    """
    plan = {key: item if not isinstance(item, (np.ndarray)) else item.tolist() for key, item in plan.items()}
    plan["transpose_forward"] = [int(i) for i in plan["transpose_forward"]]
    plan["transpose_backward"] = [int(i) for i in plan["transpose_backward"]]
    plan["do_dummy_2D_data_aug"] = bool(plan["do_dummy_2D_data_aug"])
    return plan
