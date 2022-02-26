from typing import Tuple, Union

import torch

ND_INT = Union[int, Tuple[int, int], Tuple[int, int, int]]
ND_TUPLE_INT = Union[Tuple[int, int], Tuple[int, int, int]]  # no plain int allowed


class CONV_GENERATOR:
    """
    Provides a simple wrapper around conv sequences of various combinations
    (conv -> act -> norm), pre-norm, different activations / normalisations etc.
    """

    dim: int

    def __call__(self, **kwargs) -> torch.nn.Module:
        ...
