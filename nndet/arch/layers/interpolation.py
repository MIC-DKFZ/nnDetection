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
import torch.nn.functional as F

from typing import Union, Tuple, List
from torch import Tensor


__all__ = ["InterpolateToShapes", "InterpolateToShape", "Interpolate"]


class InterpolateToShapes(torch.nn.Module):
    def __init__(self, mode: str = "nearest", align_corners: bool = None):
        """
        Downsample target tensor to size of prediction feature maps
        
        Args:
            mode:  algorithm used for upsampling: nearest, linear, bilinear,
                bicubic, trilinear, area. Defaults to "nearest".
            align_corners: Align corners points for interpolation. (see pytorch
                for more info) Defaults to None.
        
        See Also:
            :func:`torch.nn.functional.interpolate`

        Warnings:
            Use nearest for segmentation, everything else will result in
            wrong values.
        """
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners
    
    def forward(self, preds: List[Tensor], target: Tensor) -> List[Tensor]:
        """
        Interpolate target to match shape with predictions
        
        Args:
            preds: predictions to extract shape of
            target: target to interpolate
        
        Returns:
            List[Tensor]: interpolated targets
        """
        shapes = [tuple(pred.shape)[2:] for pred in preds]
        
        squeeze_result = False
        if target.ndim == preds[0].ndim - 1:
            target = target.unsqueeze(dim=1)
            squeeze_result = True

        new_targets = [F.interpolate(
            target, size=shape, mode=self.mode, align_corners=self.align_corners)
                       for shape in shapes]
        
        if squeeze_result:
            new_targets = [nt.squeeze(dim=1) for nt in new_targets]

        return new_targets


class MaxPoolToShapes(torch.nn.Module):
    def forward(self, preds: List[Tensor], target: Tensor) -> List[Tensor]:
        """
        Pool target to match shape with predictions

        Args:
            preds: predictions to extract shape of
            target: target to pool

        Returns:
            List[Tensor]: pooled targets
        """
        dim = preds[0].ndim - 2

        target_shape = list(target.shape)[-dim:]
        pool = []
        for pred in preds:
            pred_shape = list(pred.shape)[-dim:]
            pool.append(tuple([int(t / p) for t, p in zip(target_shape, pred_shape)]))

        squeeze_result = False
        if target.ndim == preds[0].ndim - 1:
            target = target.unsqueeze(dim=1)
            squeeze_result = True

        fn = getattr(F, f"max_pool{dim}d")
        new_targets = [fn(target, kernel_size=p, stride=p) for p in pool]

        if squeeze_result:
            new_targets = [nt.squeeze(dim=1) for nt in new_targets]
        return new_targets


class InterpolateToShape(InterpolateToShapes):
    """
    Interpolate predictions to target size
    """
    def forward(self, preds: List[Tensor], target: Tensor) -> List[Tensor]:
        """
        Interpolate predictions to match target

        Args:
            preds: predictions to extract shape of
            target: target to interpolate

        Returns:
            List[Tensor]: interpolated targets
        """
        shape = tuple(target.shape)[2:]

        squeeze_result = False
        if target.ndim == preds[0].ndim - 1:
            target = target.unsqueeze(dim=1)
            squeeze_result = True

        new_targets = [F.interpolate(
            pred, size=shape, mode=self.mode, align_corners=self.align_corners)
            for pred in preds]

        if squeeze_result:
            new_targets = [nt.squeeze(dim=1) for nt in new_targets]

        return new_targets


class Interpolate(torch.nn.Module):
    def __init__(self, size: Union[int, Tuple[int]] = None,
                 scale_factor: Union[float, Tuple[float]] = None,
                 mode: str = "nearest", align_corners: bool = None):
        """
        nn.Module for interpolation based on functional interpolation from 
        pytorch
        
        Args:
            size: output spatial size. Defaults to None.
            scale_factor: multiplier for spatial size. Has to match input size
                if it is a tuple. Defaults to None.
            mode:  algorithm used for upsampling: nearest, linear, bilinear,
                bicubic, trilinear, aera. Defaults to "nearest".
            align_corners: Align corners points for interpolation. (see pytorch
                for more info) Defaults to None.
        
        See Also:
            :func:`torch.nn.functional.interpolate`
        """
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Interpolate input batch
        
        Args:
            x: input tensor to interpolate
        
        Returns:
            Tensor: interpolated tensor
        """
        return F.interpolate(
            x, size=self.size, scale_factor=self.scale_factor,
            mode=self.mode, align_corners=self.align_corners)
