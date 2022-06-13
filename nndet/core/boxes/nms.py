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
from torch import Tensor
from torch.cuda.amp import autocast
from torchvision.ops.boxes import nms as nms_2d

try:
    from nndet._C import nms as nms_gpu
except ImportError:
    logger.warning("nnDetection was not build with GPU support!")
    nms_gpu = None
from nndet.core.boxes.ops import box_iou


def nms_cpu(boxes, scores, thresh):
    """
    Performs non-maximum suppression for 3d boxes on cpu
    
    Args:
        boxes (Tensor): tensor with boxes (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        scores (Tensor): score for each box [N]
        iou_threshold (float): threshould when boxes are discarded
    
    Returns:
        keep (Tensor): int64 tensor with the indices of the elements that have been kept by NMS, 
            sorted in decreasing order of scores
    """
    ious = box_iou(boxes, boxes)
    _, _idx = torch.sort(scores, descending=True)
    
    keep = []
    while _idx.nelement() > 0:
        keep.append(_idx[0])
        # get all elements that were not matched and discard all others.
        non_matches = torch.where((ious[_idx[0]][_idx] <= thresh))[0]
        _idx = _idx[non_matches]
    return torch.tensor(keep).to(boxes).long()


@autocast(enabled=False)
def nms(boxes: Tensor, scores: Tensor, iou_threshold: float):
    """
    Performs non-maximum suppression
    
    Args:
        boxes (Tensor): tensor with boxes (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        scores (Tensor): score for each box [N]
        iou_threshold (float): threshould when boxes are discarded
    
    Returns:
        keep (Tensor): int64 tensor with the indices of the elements that have been kept by NMS, 
            sorted in decreasing order of scores
    """
    if boxes.shape[1] == 4:
        # prefer torchvision in 2d because they have c++ cpu version
        nms_fn = nms_2d
    else:
        if boxes.is_cuda:
            nms_fn = nms_gpu
        else:
            nms_fn = nms_cpu
    return nms_fn(boxes.float(), scores.float(), iou_threshold)


def batched_nms(boxes: Tensor, scores: Tensor, idxs: Tensor, iou_threshold: float):
    """
    Performs non-maximum suppression in a batched fashion.
    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.
    
    Args:
        boxes (Tensor): boxes where NMS will be performed. (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        scores (Tensor): scores for each one of the boxes [N]
        idxs (Tensor): indices of the categories for each one of the boxes. [N]
        iou_threshold (float):  discards all overlapping boxes with IoU > iou_threshold
    
    Returns
        keep (Tensor): int64 tensor with the indices of the elements that have been kept by NMS, 
            sorted in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    return nms(boxes_for_nms, scores, iou_threshold)
