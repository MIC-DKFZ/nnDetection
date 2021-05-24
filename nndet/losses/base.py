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
from loguru import logger


class Loss(torch.nn.Module):
    def __init__(self,
                 *args,
                 loss_weight: float = 1.,
                 loss_fp32: bool = False,
                 reduction: str = "sum",
                 **kwargs,
                 ) -> None:
        """
        Base class for all nnDetection losses

        Args:
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            reduction: 'mean'|'sum'|'none'
                mean: mean of loss over entire batch
                sum: sum of loss over entire batch
                none: no reduction
        """
        super().__init__(*args, **kwargs)
        self.reduction = reduction
        self.loss_fp32 = loss_fp32
        self.loss_weight = loss_weight

    @property
    def loss_fp32(self) -> bool:
        return self._loss_fp32
    
    @loss_fp32.setter
    def loss_fp32(self, val: bool):
        self._loss_fp32 = val
        if val:
            logger.info(f"{self.__class__.__name__} uses FP32 loss computation.")


def reduction_helper(
    data: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """
    Helper to collapse data with different modes

    Args:
        data: data to collapse
        reduction: type of reduction. One of `mean`, `sum`, 'none'

    Returns:
        Tensor: reduced data
    """
    if reduction.lower() == 'mean':
        return torch.mean(data)
    if reduction.lower() == 'none':
        return data
    if reduction.lower() == 'sum':
        return torch.sum(data)
    raise AttributeError('Reduction parameter unknown.')
