# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0


import torch


def soft_dice(
    preds: torch.Tensor,
    targets_one_hot: torch.Tensor,
    smooth_nom: float = 0.0,
    smooth_denom: float = 1e-5,
    batch_dice: bool = False,
) -> torch.Tensor:
    """
    Compute dice loss for segmentation like feature maps

    Args:
        preds: predicted probabilities. [N, C, *], where N is the batch size,
            C is the number of classes, * are arbitrary spatial dimensions
        targets_one_hot: targets encoded as one hot. [N, C, *], where
            N is the batch size, C is the number of classes, * are
            arbitrary spatial dimensions
        smooth_nom: constant added to nominator for numerical stabilty
        smooth_denom: contant added to denominator for numerical stabilty
        batch_dice: compute statistics for each class across the whole batch
            instead of computing if per image per class

    Returns:
        torch.Tensor: compute loss. '-1' is the best possible value and '0'
            is the worst possible value
    """
    if batch_dice:
        dim = [0] + list(range(2, preds.ndim))
    else:
        dim = list(range(2, preds.ndim))

    # shapes refer to non-batch_dice version
    intersection = 2 * (preds * targets_one_hot).sum(dim=dim)  # [N, C] or [C]
    union = preds.sum(dim=dim) + targets_one_hot.sum(dim=dim)  # [N, C] or [C]
    return -1 * (intersection + smooth_nom) / (union + smooth_denom)
