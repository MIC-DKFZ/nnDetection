# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import List, Sequence, Tuple

import torch
from loguru import logger
from torch import Tensor
from torch.cuda.amp import autocast

import nndet.core.ops_torch as ops_torch
from nndet.core.rois.pooler.base import RoIPooler
from nndet.utils.typing import ND_FLOAT, ND_TUPLE_INT

try:
    from nndet._C import roi_align as roi_align_3d
except ImportError:
    logger.warning("nnDetection was not build with GPU support!")
    roi_align_3d = None


@autocast(enabled=False)
def roi_align(
    input: Tensor,
    boxes: Tensor,
    output_size: Tuple[int, int, int],
    spatial_scale: ND_FLOAT = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> Tensor:
    """
    Perfrom RoI Align accoridng to MaskRCNN paper

    Args:
        input: input feature map to extact RoI features from
            [N, C, dims], where N is the batch size, C is the number
            of channels and dims are spatial dimensions
        boxes: boxes/rois to extract features
            (batch_idx, x1, y1, x2, y2, (z1, z2))[R, dim * 2 + 1]
        output_size: output size after pooling
        spatial_scale: Define scale factor between boxes and feature
            map. Defaults to 1.0.
        sampling_ratio: Define how many points are interpolated
            in a single bin. `<=0` uses a dynamic number
            of values. Defaults to -1.
        aligned: additionally align the coordinates. Defaults to False.

    Raises:
        NotImplementedError: Currently only implemented for 3D and GPU usage

    Returns:
        Tensor: pooled features [R, C, output_size], where R is the
            number of proposal boxes, C is the number of channels and
            output_size are spatial dimensions
    """
    assert input.device == boxes.device
    if input.is_cuda:
        if boxes.shape[1] == 4:
            raise NotImplementedError
        else:
            pool_fn = roi_align_3d
    else:
        raise NotImplementedError
    boxes_pool = boxes.clone()

    # apply scaling
    if isinstance(spatial_scale, Sequence):
        _scale = torch.tensor(spatial_scale, dtype=boxes.dtype, device=boxes.device)
        boxes_pool[:, 1:] = boxes_pool[:, 1:] * ops_torch.expand_to_boxes(_scale)
    else:
        boxes_pool[:, 1:] = boxes_pool[:, 1:] * spatial_scale
    spatial_scale = 1.0

    if aligned:
        boxes_pool[:, 1:] = boxes_pool[:, 1:] - 0.5

    if input.is_cuda and input.dtype != boxes.dtype:
        # use float32 for pooling
        input = input.float()
        boxes_pool = boxes_pool.float()

    res = pool_fn(
        input.contiguous(),
        boxes_pool.contiguous(),
        spatial_scale,
        output_size[0],
        output_size[1],
        output_size[2],
        sampling_ratio,
    )
    return res


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
        Pool features via RoI Align

        Args:
            fmap: feature map to pool form [N, C, dims] where N is the
                batch size, C is the number of channels and dims are
                spatial dimensions
            proposal_boxes_batch_idx: proposal boxes with batch index inserted
                in the first channel
                (batch_idx, x1, y1, x2, y2, (z1, z2))[R, dim * 2 + 1]
            spatial_scale: the ratio of the size of the feature map and the
                original image (always <= 1)

        Returns:
            Tensor: pooled features from feature map [R, C, output_size]
                where R is the number of proposal boxes, C is the number
                channels and output_size are spatial dimensions
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
                C is the number of instances, sdims are spatial dimensions
            proposal_boxes: proposal boxes to pool
                (x1, y1, x2, y2, (z1, z2))[R, dim * 2]
            matched_gt_idx: index of matched ground truth box. The n-th
                box needs to correspond to the n-th channel inside the
                binary segmentation mask

        Returns:
            List[Tensor]: pooled masks [R, output_size], where R
                is the number of proposal boxes and output_size are
                spatial dimensions
        """
        output_size = self.feature_output_size if self.mask_output_size is None else self.mask_output_size

        pooled_masks = []
        assert len(binary_masks) == len(proposal_boxes)
        assert len(binary_masks) == len(matched_gt_idx)
        for m, p_boxes, m_idx in zip(binary_masks, proposal_boxes, matched_gt_idx):
            if m.numel() == 0 or p_boxes.numel() == 0:
                # no ground truth in batch => can not compute mask loss on roi with FG
                _pooled_masks_image = torch.tensor(
                    [],
                    dtype=p_boxes.dtype,
                    device=p_boxes.device,
                ).view(0, *output_size)
            else:
                p_boxes_batch_idx = torch.cat([m_idx[:, None], p_boxes], dim=1)
                _pooled_masks_image = roi_align(
                    input=m[:, None],
                    boxes=p_boxes_batch_idx,
                    output_size=output_size,
                    spatial_scale=1.0,
                    **self.mask_pool_kwargs,
                )[
                    :, 0
                ]  # truncate artifically added channel dimension
            pooled_masks.append(_pooled_masks_image)
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

        Args:
            proposal_boxes: proposal boxes
                (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            features: feature maps which should be used for pooling
                from backbone/fpn each with [N, C, dims]
                Ordered from highest resolution feature map (0)
                to the lowed resolution one (-1).
            image_size: spatial size of image

        Returns:
            Tensor: level for each proposal [N]
        """
        image_size_tensor = torch.tensor(
            image_size,
            dtype=proposal_boxes.dtype,
            device=proposal_boxes.device,
        )
        proposal_boxes_norm = proposal_boxes / ops_torch.expand_to_boxes(image_size_tensor)

        _, d2, d3 = ops_torch.box_size(proposal_boxes_norm).unbind(dim=-1)

        num_levels = len(features)
        level = (
            (num_levels + torch.log2(torch.sqrt(d2 * d3))).round().clamp_(min=0, max=num_levels).to(dtype=torch.long)
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
        Assign proposals to pyramid levels for pooling. Similar to
        `RoIAlignOrigAssign` but includes the depth dimension

        Args:
            proposal_boxes: proposal boxes
                (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            features: feature maps which should be used for pooling
                from backbone/fpn each with [N, C, dims]
                Ordered from highest resolution feature map (0)
                to the lowed resolution one (-1).
            image_size: spatial size of image

        Returns:
            Tensor: level for each proposal [N]
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
        proposal_boxes_norm = (proposal_boxes * 1.33) / ops_torch.expand_to_boxes(image_size_tensor)
        normed_size = ops_torch.box_size(proposal_boxes_norm)  # [N, 3]

        if len(image_size) == 2:
            v = torch.log2((normed_size[:, 0] * normed_size[:, 1]).sqrt())
        elif len(image_size) == 3:
            v = torch.log2((normed_size[:, 0] * normed_size[:, 1] * normed_size[:, 2]) ** (1 / 3))
        else:
            raise ValueError(f"Image size needs to be 2D or 3d, received {image_size}.")

        level = (v + num_levels).clamp_(min=0, max=num_levels).to(dtype=torch.long)
        return level
