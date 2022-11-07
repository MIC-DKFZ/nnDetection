# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from torchvision (https://github.com/pytorch/vision) licensed under
# SPDX-FileCopyrightText: Soumith Chintala 2016
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Tuple

import torch

from nndet.core.boxes.matcher import MatcherType


def assign_targets_to_anchors(
    proposal_matcher: MatcherType,
    anchors: List[torch.Tensor],
    target_boxes: List[torch.Tensor],
    target_classes: List[torch.Tensor],
    **kwargs,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Compute labels and matched ground truth for each anchor
    Adapted from torchvision https://github.com/pytorch/vision

    Args:
        anchors: anchors *per image* List[[N, dim * 2]],
            N=number of anchors per image
        target_boxes: ground truth boxes *per image* List[[X, dim * 2]],
            X=number of gt per image
        target_classes: ground truth classes *per image* (classes start from 0)
            List[[X]], X=number of gt per image
        kwargs: keyword arguments passed to anchor matcher

    Returns:
        List[torch.Tensor]: labels ([1, K]: foreground classes, 0: background,
            -1: between) List[[N]], N=number of anchors per image
        List[torch.Tensor]: matched gt box List[[N, dim *  2]],
            N=number of anchors per image
        List[Tensor]: vector which contains the matched box index for all
            anchors (if background `BELOW_LOW_THRESHOLD` is used
            and if it should be ignored `BETWEEN_THRESHOLDS` is used) [N]
    """
    labels = []
    matched_gt_boxes = []
    matched_idx_list = []
    for anchors_per_image, gt_boxes, gt_classes in zip(anchors, target_boxes, target_classes):
        # indices of ground truth box for each proposal
        match_quality_matrix, matched_idxs = proposal_matcher(
            gt_boxes,
            anchors_per_image,
            **kwargs,
        )

        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        if match_quality_matrix.numel() > 0:
            matched_idxs_clamp = matched_idxs.clamp(min=0)
            matched_gt_boxes_per_image = gt_boxes[matched_idxs_clamp]

            # Positive (negative indices can be ignored because they are overwritten in the next step)
            # this influences how background class is handled in the input!!!! (here +1 for background)
            labels_per_image = gt_classes[matched_idxs_clamp].to(dtype=anchors_per_image.dtype)
            labels_per_image = labels_per_image + 1
        else:
            num_anchors_per_image = anchors_per_image.shape[0]
            # no ground truth => no matches, all background
            matched_gt_boxes_per_image = torch.zeros_like(anchors_per_image)
            labels_per_image = torch.zeros(num_anchors_per_image).to(anchors_per_image)

        # Background (negative examples)
        bg_indices = matched_idxs == proposal_matcher.BELOW_LOW_THRESHOLD
        labels_per_image[bg_indices] = 0.0

        # discard indices that are between thresholds
        inds_to_discard = matched_idxs == proposal_matcher.BETWEEN_THRESHOLDS
        labels_per_image[inds_to_discard] = -1.0

        labels.append(labels_per_image)
        matched_gt_boxes.append(matched_gt_boxes_per_image)
        matched_idx_list.append(matched_idxs)
    return labels, matched_gt_boxes, matched_idx_list
