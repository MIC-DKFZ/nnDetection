# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from torchvision (https://github.com/pytorch/vision) licensed under
# SPDX-FileCopyrightText: Soumith Chintala 2016
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import division

import math
from typing import Sequence, TypeVar

import torch
from torch.jit.annotations import List, Tuple
from torch import Tensor

from torchvision.models.detection._utils import BoxCoder


@torch.jit.script
def encode_boxes(reference_boxes: torch.Tensor,
                 proposals: torch.Tensor,
                 weights: torch.Tensor,
                 ) -> torch.Tensor:
    """
    Encode a set of proposals with respect to some reference boxes

    Args:
        reference_boxes: reference boxes (x1, y1, x2, y2, (z1, z2))
        proposals: boxes to be encoded (x1, y1, x2, y2, (z1, z2))
        weights: weights for dimensions (wx, wy, ww, wh, wz, wd)
    """
    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # implementation starts here
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    if proposals.shape[1] == 6:
        wz = weights[4]
        wd = weights[5]

        proposals_z1 = proposals[:, 4].unsqueeze(1)
        proposals_z2 = proposals[:, 5].unsqueeze(1)
        ex_depth = proposals_z2 - proposals_z1
        ex_ctr_z = proposals_z1 + 0.5 * ex_depth

        reference_boxes_z1 = reference_boxes[:, 4].unsqueeze(1)
        reference_boxes_z2 = reference_boxes[:, 5].unsqueeze(1)
        gt_depth = reference_boxes_z2 - reference_boxes_z1
        gt_ctr_z = reference_boxes_z1 + 0.5 * gt_depth

        targets_dz = wz * (gt_ctr_z - ex_ctr_z) / ex_depth
        targets_dd = wd * torch.log(gt_depth / ex_depth)

        targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh,
                             targets_dz, targets_dd), dim=1)
    else:
        targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


def decode_single(rel_codes: Tensor, boxes: Tensor,
                  weights: Sequence[float],
                  bbox_xform_clip: float) -> Tensor:
    """
    From a set of original boxes and encoded relative box offsets,
    get the decoded boxes.

    Args:
        rel_codes: encoded boxes [Num_boxes x (dim * 2)] (dx, dy, dw, dh, dz, dd)
        boxes: reference boxes (x1, y1, x2, y2, (z1, z2))
    """
    # offset is 4 in case of 2d data and 6 in case of 3d
    offset = boxes.shape[1]
    boxes = boxes.to(rel_codes.dtype)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    dx = rel_codes[:, 0::offset] / wx
    dy = rel_codes[:, 1::offset] / wy
    dw = rel_codes[:, 2::offset] / ww
    dh = rel_codes[:, 3::offset] / wh

    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=bbox_xform_clip)
    dh = torch.clamp(dh, max=bbox_xform_clip)

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    pred_boxes1 = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype) * pred_w
    pred_boxes2 = pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_y.dtype) * pred_h
    pred_boxes3 = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype) * pred_w
    pred_boxes4 = pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_y.dtype) * pred_h

    if offset == 6:
        depths = boxes[:, 5] - boxes[:, 4]
        ctr_z = boxes[:, 4] + 0.5 * depths

        wz = weights[4]
        wd = weights[5]

        dz = rel_codes[:, 4::offset] / wz
        dd = rel_codes[:, 5::offset] / wd
        dd = torch.clamp(dd, max=bbox_xform_clip)

        pred_ctr_z = dz * depths[:, None] + ctr_z[:, None]
        pred_z = torch.exp(dd) * depths[:, None]

        pred_boxes5 = pred_ctr_z - torch.tensor(0.5, dtype=pred_ctr_z.dtype) * pred_z
        pred_boxes6 = pred_ctr_z + torch.tensor(0.5, dtype=pred_ctr_z.dtype) * pred_z
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4,
                                  pred_boxes5, pred_boxes6), dim=2).flatten(1)
    else:
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4),
                                 dim=2).flatten(1)
    return pred_boxes


class BoxCoderND(BoxCoder):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    Compatible with 2d and 3d
    """
    def encode(self,
               reference_boxes: List[Tensor], 
               proposals: List[Tensor],
               ) -> Tuple[Tensor]:
        """
        Encode a set of proposals with respect to some reference boxes

        Args:
            reference_boxes: reference boxes for each image.
                (x1, y1, x2, y2, (z1, z2))
            proposals: proposals for each image
                (x1, y1, x2, y2, (z1, z2))

        Returns:
            Tuple[Tensor]: regression targets for each image
        """
        # filter for images which have a foreground class
        filter_min_one_gt = [rb.numel() > 0 for rb in reference_boxes]
        filtered_ref_boxes = [
            rb for idx, rb in enumerate(reference_boxes) if filter_min_one_gt[idx]]
        filtered_proposals = [
            pr for idx, pr in enumerate(proposals) if filter_min_one_gt[idx]]

        if any(filter_min_one_gt):
            filtered_encoded = super().encode(filtered_ref_boxes, filtered_proposals)

        # fill image with no ground truth
        idx_enc = 0
        encoded = []
        for img_idx, gt_present in enumerate(filter_min_one_gt):
            if gt_present:
                encoded.append(filtered_encoded[idx_enc])
                idx_enc += 1
            else:
                # fill with zeros because they  do not contribute to the
                # regression loss anyway (all anchors are labeled as background)
                encoded.append(torch.zeros_like(proposals[img_idx]))
        return encoded

    def encode_single(self,
                      reference_boxes: Tensor,
                      proposals: Tensor,
                      ) -> Tensor:
        """
        Encode a set of proposals with respect to some reference boxes

        Arguments:
            reference_boxes: reference boxes  (x1, y1, x2, y2, (z1, z2))
            proposals: boxes to be encoded  (x1, y1, x2, y2, (z1, z2))
        """
        dtype, device = reference_boxes.dtype, reference_boxes.device
        weights = torch.tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)
        return targets

    def decode(self, rel_codes: Tensor, boxes: List[Tensor]) -> Tensor:
        """
        Decode boxes

        Args:
            rel_codes: relative offsets to reference boxes 
                (dx, dy, dw, dh, (dz, dd))[N, dim * 2]
            boxes: list of reference boxes per image
                (x1, y1, x2, y2, (z1, z2))

        Returns:
            Tensor: decoded boxes
        """
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        spatial_dims = concat_boxes.shape[1]
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        pred_boxes = self.decode_single(rel_codes.reshape(box_sum, -1), concat_boxes)
        return pred_boxes.reshape(box_sum, spatial_dims)

    def decode_single(self, rel_codes: torch.Tensor, boxes: torch.Tensor):
        dtype, device = rel_codes.dtype, rel_codes.device
        return decode_single(rel_codes, boxes, self.weights, self.bbox_xform_clip)


CoderType = TypeVar('CoderType', bound=BoxCoderND)
