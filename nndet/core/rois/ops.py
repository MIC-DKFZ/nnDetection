from typing import List, Optional, Sequence, Union

import torch

from nndet.utils.tensor import cat


def create_binary_masks(
    mask: torch.Tensor,
    num_instances: Optional[Sequence[int]] = None,
) -> List[torch.Tensor]:
    """
    Create binary masks from numbered mask

    Args:
        mask: mask with *consecutively* numbered instances
        num_instances: save some compute by providing the number
            of instances per image

    Returns:
        List[torch.Tensor]: binary masks List[[X, dims]]
    """
    masks = []
    for i, m in enumerate(mask.split(split_size=1, dim=0)):
        # FIXME
        unique_ids = m.unique()
        masks.append(cat([(m == ui).to(m) for ui in unique_ids], dim=0)[1:])

        # _m = m[0]  # remove channel dims
        # if num_instances is not None:
        #     ni = num_instances[i]
        # else:
        #     ni = m.max()
        # print(f"max: {_m.max()} unique: {_m.unique()} num_instances: {num_instances}")
        # masks.append(
        #     torch.zeros(size=(int(ni) + 1, *_m.shape), device=_m.device).scatter_(
        #         0, _m.long().unsqueeze(0), 1.0
        #     )[1:]
        # )
    return masks


# TODO: heck output shape, channel location
def binary_masks_to_seg(
    binary_masks: List[torch.Tensor],
    gt_classes: List[torch.Tensor],
) -> Union[List[torch.Tensor], torch.Tensor]:
    """
    Convert binary masks to semantic segmentation

    Args:
        binary_masks: list of binary masks List[[X, dims]]
        gt_classes: list of ground truth classes List[[X]]

    Returns:
        List[torch.Tensor]: semantic segmentation. [N, dims]

    Warnings:
        Only supports non overlapping binary masks.
    """
    assert len(binary_masks) == len(gt_classes)
    return torch.stack(
        [(bn * gtc).max(dim=0) for bn, gtc in zip(binary_masks, gt_classes)], dim=0
    )
