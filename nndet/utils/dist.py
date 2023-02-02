# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Original code from DETR httpshttps://github.com/facebookresearch/detrgithub.com/facebookresearch/detr/blob/main/models/matcher.py  # noqa: E501
# SPDX-FileCopyrightText: 2020 Facebook
# SPDX-License-Identifier: Apache-2.0

import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    """
    Helper function to check if distributed training is used

    Returns:
        _type_: _description_
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    """
    Helper function to obtain world size
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()
