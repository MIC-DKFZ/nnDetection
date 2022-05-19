# from dataclasses import dataclass
from typing import Dict, Union

import torch

# @dataclass
# class BodyInfo:

#     strides:
#     kernels:

#     Dict[str, ]

#     channels: Dict[int]
#     strides: Sequence[Union[Sequence[int], int]]

# TODO: how to  unify information?
# Relative to Absolute Stride conversion
# unified kernel + stride format
# padding for kernels
# Norm & Act ?


class BodyOutput(torch.nn.Module):
    def __init__(self, output: Dict[str, torch.Tensor]) -> None:
        """
        Defined a standardized format to work with output from the different
        bodyparts of the network (e.g. backbone, neck)

        Args:
            output: output which should be wrapped by this container. The
                container needs to be named in the format {P[X]: tensor}
                where X is an integer ranging from 0 (highest resolution)
                to N (lowest resolution)
        """
        super().__init__()

        output_std = {}
        for key, item in output.items():
            if isinstance(key, int):
                output_std[f"P{key}"] = item
            else:
                assert key.startswith("P")
                output_std[key] = item
        self.output = torch.ParameterDict(output)

    def __getitem__(self, key: Union[int, str]) -> torch.Tensor:
        """
        Access individual element

        Args:
            key: return element. If int, it will automatically formatted
                to the P[X] format

        Returns:
            torch.Tensor: selecetd element
        """
        if isinstance(key, int):
            return self.output[f"P{key}"]
        else:
            return self.output[key]

    def first_level(self) -> int:
        """
        First present level (inclusive)

        Returns:
            int: first level
        """
        levels = [int(p[1:]) for p in self.output_std.keys()]
        return min(levels)

    def last_level(self) -> int:
        """
        Last present level (inclusive)

        Returns:
            int: last level
        """
        levels = [int(p[1:]) for p in self.output_std.keys()]
        return max(levels)

    def num_levels(self) -> int:
        """
        Number of levels

        Returns:
            int: number of levels
        """
        return len(self.output_std)
