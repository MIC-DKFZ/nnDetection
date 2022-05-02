import numpy as np


def bin_mask_iou_np(
    bin_masks1: np.ndarray,
    bin_masks2: np.ndarray,
) -> np.ndarray:
    bin_masks1_flattened = bin_masks1.reshape(bin_masks1.shape[0], -1)
    bin_masks2_flattened = bin_masks2.reshape(bin_masks2.shape[0], -1)

    masks1_vol = bin_masks1_flattened.sum(axis=1)  # [N]
    masks2_vol = bin_masks2_flattened.sum(axis=1)  # [M]

    intersection = np.matmul(bin_masks1_flattened, bin_masks2_flattened.T)  # [N, M]
    union = masks1_vol[:, None] + masks2_vol[None] - intersection  # [N, M]
    return intersection / union
