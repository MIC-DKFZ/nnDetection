"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch

__all__ = ["reduction_helper"]


def reduction_helper(data: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    Helper to collapse data with different modes

    Args:
        data: data to collapse
        reduction: type of reduction. One of `mean`, `sum`, None

    Returns:
        Tensor: reduced data
    """
    if reduction == 'mean':
        return torch.mean(data)
    if reduction == 'none' or reduction is None:
        return data
    if reduction == 'sum':
        return torch.sum(data)
    raise AttributeError('Reduction parameter unknown.')
