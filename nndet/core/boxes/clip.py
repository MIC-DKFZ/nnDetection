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

from typing import Tuple

import torch


def clip_boxes_to_image_(boxes: torch.Tensor, img_shape: Tuple[int]):
    """
    Clip boxes to image dimensions inplace

    Args:
        boxes (Tensor): tensor with boxes [N x (2*dim)] (x_min, y_min, x_max, y_max(, z_min, z_max))
        img_shape (Tuple[height, width(, depth)]): size of image

    Returns:
        Tensor: clipped boxes as tensor

    Raises:
        ValueError: boxes need to have 4(2D) or 6(3D) components
    """
    if boxes.shape[-1] == 4:
        return clip_boxes_to_image_2d_(boxes, img_shape)
    elif boxes.shape[-1] == 6:
        return clip_boxes_to_image_3d_(boxes, img_shape)
    else:
        raise ValueError(f"Boxes with {boxes.shape[-1]} are not supported.")


def clip_boxes_to_image(boxes: torch.Tensor, img_shape: Tuple[int]):
    """
    Clip boxes to image dimensions

    Args:
        boxes (Tensor): tensor with boxes [N x (2*dim)] (x_min, y_min, x_max, y_max(, z_min, z_max))
        img_shape (Tuple[height, width(, depth)]): size of image

    Returns:
        Tensor: clipped boxes as tensor

    Raises:
        ValueError: boxes need to have 4(2D) or 6(3D) components
    """
    if boxes.shape[-1] == 4:
        return clip_boxes_to_image_2d(boxes, img_shape)
    elif boxes.shape[-1] == 6:
        return clip_boxes_to_image_3d(boxes, img_shape)
    else:
        raise ValueError(f"Boxes with {boxes.shape[-1]} are not supported.")


def clip_boxes_to_image_2d_(boxes: torch.Tensor, img_shape: Tuple[int, int]):
    """
    Clip boxes to image dimensions

    Args:
        boxes (Tensor): tensor with boxes [N x 4] (x_min, y_min, x_max, y_max)
        img_shape (Tuple[x_max, y_max]): size of image

    Returns:
        Tensor: clipped boxes as tensor
    """
    s0, s1 = img_shape
    boxes[..., 0::2].clamp_(min=0, max=s0)
    boxes[..., 1::2].clamp_(min=0, max=s1)
    return boxes


def clip_boxes_to_image_3d_(boxes: torch.Tensor, img_shape: Tuple[int, int, int]):
    """
    Clip boxes to image dimensions

    Args:
        boxes (Tensor): tensor with boxes [N x 6] (x_min, y_min, x_max, y_max, z_min, z_max)
        img_shape (Tuple[height, width, depth]): size of image

    Returns:
        Tensor: clipped boxes as tensor
    """
    s0, s1, s2 = img_shape
    boxes[..., 0::6].clamp_(min=0, max=s0)
    boxes[..., 1::6].clamp_(min=0, max=s1)
    boxes[..., 2::6].clamp_(min=0, max=s0)
    boxes[..., 3::6].clamp_(min=0, max=s1)
    boxes[..., 4::6].clamp_(min=0, max=s2)
    boxes[..., 5::6].clamp_(min=0, max=s2)
    return boxes


def clip_boxes_to_image_2d(boxes: torch.Tensor, img_shape: Tuple[int, int]):
    """
    Clip boxes to image dimensions

    Args:
        boxes (Tensor): tensor with boxes [N x 4] (x_min, y_min, x_max, y_max)
        img_shape (Tuple[x_max, y_max]): size of image

    Returns:
        Tensor: clipped boxes as tensor

    Notes:
        Uses float32 internally because clipping of half cpu tensors is not
        supported
    """
    s0, s1 = img_shape
    boxes[..., 0::2] = boxes[..., 0::2].clamp(min=0, max=s0)
    boxes[..., 1::2] = boxes[..., 1::2].clamp(min=0, max=s1)
    return boxes


def clip_boxes_to_image_3d(boxes: torch.Tensor, img_shape: Tuple[int, int, int]):
    """
    Clip boxes to image dimensions

    Args:
        boxes (Tensor): tensor with boxes [N x 6] (x_min, y_min, x_max, y_max, z_min, z_max)
        img_shape (Tuple[height, width, depth]): size of image

    Returns:
        Tensor: clipped boxes as tensor

    Notes:
        Uses float32 internally because clipping of half cpu tensors is not
        supported
    """
    s0, s1, s2 = img_shape
    boxes[..., 0::6] = boxes[..., 0::6].clamp(min=0, max=s0)
    boxes[..., 1::6] = boxes[..., 1::6].clamp(min=0, max=s1)
    boxes[..., 2::6] = boxes[..., 2::6].clamp(min=0, max=s0)
    boxes[..., 3::6] = boxes[..., 3::6].clamp(min=0, max=s1)
    boxes[..., 4::6] = boxes[..., 4::6].clamp(min=0, max=s2)
    boxes[..., 5::6] = boxes[..., 5::6].clamp(min=0, max=s2)
    return boxes
