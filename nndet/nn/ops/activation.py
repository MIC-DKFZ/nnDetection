# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from loguru import logger


# @torch.compile
def swish(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """
    Apples swish function as described in
    https://arxiv.org/abs/1710.05941
    f(x) = x * sigmoid(x)

    Args:
        x: input tensor
        inplace: optionally performs this operation inplace (
            NOT implemented here)
    """
    return x.mul(x.sigmoid())


class Swish(torch.nn.Module):
    def __init__(self, inplace: bool = False):
        """
        Apples swish function as described in
        https://arxiv.org/abs/1710.05941
        f(x) = x * sigmoid(x)

        Args:
            inplace: optionally performs this operation inplace (
                NOT implemented here)
        """
        super().__init__()
        self.inplace = inplace
        if self.inplace:
            logger.warning("Inplace not implemented for Swish activation")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish(x, inplace=self.inplace)


# @torch.compile
def mish(
    x: torch.Tensor,
    inplace: bool = False,
    beta: float = 1,
    threshold: float = 20,
) -> torch.Tensor:
    """
    Apples mish function as described in
    https://www.bmvc2020-conference.com/assets/papers/0928.pdf
    f(x) = x * tanh(softplus(x))

    Args:
        x: input tensor
        inplace: optionally performs this operation inplace (
            NOT implemented here)
        beta: see pytorch softplus docu
        threshold: see pytorch softplus docu
    """
    return x.mul(F.softplus(x, beta=beta, threshold=threshold).tanh())


class Mish(torch.nn.Module):
    def __init__(
        self,
        inplace: bool = False,
        beta: float = 1,
        threshold: float = 20,
    ):
        """
        Apples mish function as described in
        https://www.bmvc2020-conference.com/assets/papers/0928.pdf
        f(x) = x * tanh(softplus(x))

        Args:
            inplace: optionally performs this operation inplace (
                NOT implemented here)
            beta: see pytorch softplus docu
            threshold: see pytorch softplus docu
        """
        super().__init__()
        self.inplace = inplace
        if self.inplace:
            logger.warning("Inplace not supported for Mish.")
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mish(
            x,
            inplace=self.inplace,
            beta=self.beta,
            threshold=self.threshold,
        )
