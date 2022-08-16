# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import List, Sequence, Tuple

import torch
from loguru import logger
from torch import Tensor

from nndet.core.boxes.ops import box_size, expand_to_boxes
from nndet.core.rois.pooler.base import RoIPooler
from nndet.utils.typing import ND_FLOAT, ND_TUPLE_INT

try:
    from nndet._C import roi_align as roi_align_3d
except ImportError:
    logger.warning("nnDetection was not build with GPU support!")
    roi_align_3d = None


def roi_align(
    input: Tensor,
    boxes: Tensor,
    output_size: Tuple[int],
    spatial_scale: ND_FLOAT = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> Tensor:
    assert input.device == boxes.device

    # apply scaling here, will be moved to cuda function down the road
    if isinstance(spatial_scale, Sequence):
        _scale = torch.tensor(spatial_scale, dtype=boxes.dtype, device=boxes.device)
        boxes[:, 1:] = boxes[:, 1:] * expand_to_boxes(_scale)
    else:
        boxes[:, 1:] = boxes[:, 1:] * spatial_scale
    spatial_scale = 1.0

    if input.is_cuda:
        if boxes.shape[1] == 4:
            raise NotImplementedError
        else:
            pool_fn = roi_align_3d
    else:
        raise NotImplementedError

    boxes = boxes.to(dtype=input.dtype)

    # print(boxes)
    return pool_fn(
        input.contiguous(),
        boxes.contiguous(),
        spatial_scale,
        output_size[0],
        output_size[1],
        output_size[2],
        sampling_ratio,
    )


class RoIAlignBase(RoIPooler):
    """
    Define Ops with RoI Align
    """

    def _pool_features(
        self,
        fmap: torch.Tensor,
        proposal_boxes_batch_idx: torch.Tensor,
        spatial_scale: ND_FLOAT,
    ) -> torch.Tensor:
        """
        Pooling feature for proposals from given feature map
        """
        return roi_align(
            input=fmap,
            boxes=proposal_boxes_batch_idx.detach(),
            output_size=self.feature_output_size,
            spatial_scale=spatial_scale,
            **self.feature_pool_kwargs,
        )

    @torch.no_grad()
    def pool_masks(
        self,
        binary_masks: List[Tensor],
        proposal_boxes: List[Tensor],
        matched_gt_idx: List[Tensor],
    ) -> List[Tensor]:
        """
        Pooling masks for given matched gt boxes

        Args:
            binary_masks: binary segmentation masks [C, sdims]
                C=number of instances
            proposal_boxes: proposal boxes to pool
                (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            matched_gt_idx: index of matched ground truth box. The n-th
                box needs to correspond to the n-th channel inside the
                binary segmentation mask

        Returns:
            Tensor: pooled masks [N, output_size]
        """
        output_size = (
            self.feature_output_size
            if self.mask_output_size is None
            else self.mask_output_size
        )

        pooled_masks = []
        for m, p_boxes, m_idx in zip(binary_masks, proposal_boxes, matched_gt_idx):
            p_boxes_batch_idx = torch.cat([m_idx[:, None], p_boxes], dim=1)
            if m.numel() == 0:
                # no ground truth
                pooled_masks.append(
                    torch.tensor(
                        [],
                        dtype=p_boxes.dtype,
                        device=p_boxes.device,
                    ).view(0, *output_size)
                )  # empty mask with correct shape for concatenation
            else:
                pooled_masks.append(
                    roi_align(
                        input=m[:, None],
                        boxes=p_boxes_batch_idx,
                        output_size=output_size,
                        spatial_scale=1.0,
                        **self.mask_pool_kwargs,
                    )[:, 0]
                )
        return pooled_masks


class RoIAlignOrigAssign(RoIAlignBase):
    @torch.no_grad()
    def _find_pyramid_level(
        self,
        proposal_boxes: torch.Tensor,
        features: List[torch.Tensor],
        image_size: ND_TUPLE_INT,
    ) -> torch.Tensor:
        """
        Assign proposals to pyramid levels for pooling

        This is equivalent to the original in
        `Feature Pyramid Networks for Object Detection`
        https://arxiv.org/pdf/1612.03144.pdf and MDT

        => this ignores the z axes completely
        """
        image_size_tensor = torch.tensor(
            image_size,
            dtype=proposal_boxes.dtype,
            device=proposal_boxes.device,
        )
        proposal_boxes_norm = proposal_boxes / expand_to_boxes(image_size_tensor)

        _, d2, d3 = box_size(proposal_boxes_norm).unbind(dim=-1)

        num_levels = len(features)
        level = (
            (num_levels + torch.log2(torch.sqrt(d2 * d3)))
            .round()
            .clamp_(min=0, max=num_levels)
            .to(dtype=torch.long)
        )
        return level


class RoIAlignNaiveAssign(RoIAlignBase):
    """
    Define assignment V1
    """

    @torch.no_grad()
    def _find_pyramid_level(
        self,
        proposal_boxes: torch.Tensor,
        features: List[torch.Tensor],
        image_size: ND_TUPLE_INT,
    ) -> torch.Tensor:
        """
        Assign proposals to pyramid levels for pooling
        """
        num_levels = len(features)
        image_size_tensor = torch.tensor(
            image_size,
            dtype=proposal_boxes.dtype,
            device=proposal_boxes.device,
        )

        # We normalize the box size instead of the area/vol
        # since this should give better numerical results especially
        # when using mixed precision (i.e. 128^3 does not fit float16)
        proposal_boxes_norm = (proposal_boxes * 1.33) / expand_to_boxes(
            image_size_tensor
        )
        normed_size = box_size(proposal_boxes_norm)  # [N, 3]

        if len(image_size) == 2:
            v = torch.log2((normed_size[:, 0] * normed_size[:, 1]).sqrt())
        elif len(image_size) == 3:
            v = torch.log2(
                (normed_size[:, 0] * normed_size[:, 1] * normed_size[:, 2]) ** (1 / 3)
            )
        else:
            raise ValueError(f"Image size needs to be 2D or 3d, received {image_size}.")

        level = (v + num_levels).clamp_(min=0, max=len(features)).to(dtype=torch.long)
        return level
