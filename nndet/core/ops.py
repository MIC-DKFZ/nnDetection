# Modifications licensed under
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

# ivnerse_sigmoid function from
# https://github.com/fundamentalvision/Deformable-DETR/blob/11169a60c33333af00a4849f1808023eba96a931/util/misc.py  # noqa: E501
# SPDX-FileCopyrightText: 2020 SenseTime
# SPDX-License-Identifier: Apache-2.0

import torch


def inverse_sigmoid(data: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Inverse Sigmoid Function for pre-sigmoid additions

    Args:
        data: input tensor
        eps: epsilon for numerical stability

    Returns:
        torch.Tensor: inverse sigmoid of values in original tensor
    """
    data = data.clamp(min=0, max=1)
    x1 = data.clamp(min=eps)
    x2 = (1 - data).clamp(min=eps)
    return torch.log(x1 / x2)
