# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from nndet.utils.typing import ND_FLOAT, ND_TUPLE_INT


class RoIPooler(torch.nn.Module):
    def __init__(
        self,
        feature_output_size: ND_TUPLE_INT,
        mask_output_size: Optional[ND_TUPLE_INT] = None,
        feature_pool_kwargs: Optional[Dict] = None,
        mask_pool_kwargs: Optional[Dict] = None,
    ):
        """
        Class to perform RoI Pooling for multi scale features and optionally
        masks

        Args:
            feature_output_size: spatial output size of features produced by
                pooling operation
            mask_output_size: spatial output size of masks produced by
                pooling operation
            feature_pool_kwargs: keyword arguments passed to feature pooling
                operation
            mask_pool_kwargs: keyword arguments passed to mask pooling
                operation
        """
        super().__init__()
        self.feature_output_size = feature_output_size
        self.mask_output_size = mask_output_size
        self.feature_pool_kwargs = feature_pool_kwargs if feature_pool_kwargs is not None else {}
        self.mask_pool_kwargs = mask_pool_kwargs if mask_pool_kwargs is not None else {}

    def forward(
        self,
        features: List[torch.Tensor],
        proposal_boxes: torch.Tensor,
        batch_idx: torch.Tensor,
        image_size: ND_TUPLE_INT,
    ) -> torch.Tensor:
        """
        Perform multiscale pyramid pooling

        Args:
            features: feature maps which should be used for pooling
                from backbone/fpn, each with [N, C, dims]
                Ordered from highest resolution feature map (0)
                to the lowed resolution one (-1).
            proposal_boxes: box proposals
                (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            batch_idx: original batch index of each proposal [N]
            image_size: spatial size of image

        Returns:
            Tensor: extracted features [N, C, output_size]
        """
        proposal_boxes_batch_idx = torch.cat(
            [batch_idx[:, None], proposal_boxes],
            dim=1,
        )  # [N, dim * 2 + 1]

        assert len(image_size) + 2 == features[0].ndim

        if len(features) == 1:
            spatial_scale = tuple(features[0].shape[i + 2] / image_size[i] for i in range(len(image_size)))
            output = self._pool_features(
                fmap=features[0],
                proposal_boxes_batch_idx=proposal_boxes_batch_idx,
                spatial_scale=spatial_scale,
            ).to(dtype=features[0].dtype)
        else:
            # determine level dynamically
            proposal_levels = self._find_pyramid_level(
                features=features,
                proposal_boxes=proposal_boxes,
                image_size=image_size,
            )

            output = torch.zeros(
                [
                    proposal_boxes_batch_idx.shape[0],
                    features[0].shape[1],
                    *self.feature_output_size,
                ],
                dtype=features[0].dtype,
                device=features[0].device,
            )

            for idx, fmap in enumerate(features):
                spatial_scale = tuple(fmap.shape[i + 2] / image_size[i] for i in range(len(image_size)))
                idx = torch.where(proposal_levels == idx)[0]
                if idx.numel() > 0:
                    output[idx] = self._pool_features(
                        fmap=fmap,
                        proposal_boxes_batch_idx=proposal_boxes_batch_idx[idx],
                        spatial_scale=spatial_scale,
                    ).to(dtype=output.dtype)
        return output

    @abstractmethod
    @torch.no_grad()
    def _find_pyramid_level(
        self,
        features: List[torch.Tensor],
        proposal_boxes: torch.Tensor,
        image_size: Union[Tuple[int, int], Tuple[int, int, int]],
    ) -> torch.Tensor:
        """
        Assign proposals to pyramid levels for pooling

        Args:
            proposal_boxes: proposal boxes
                (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            features: feature maps which should be used for pooling
                from backbone/fpn
            image_size: spatial size of image

        Returns:
            Tensor: Index of pyramid level for each for pooling
        """
        raise NotImplementedError

    @abstractmethod
    def _pool_features(
        self,
        fmap: torch.Tensor,
        proposal_boxes_batch_idx: torch.Tensor,
        spatial_scale: ND_FLOAT,
    ) -> torch.Tensor:
        """
        Pooling feature for proposals from given feature map

        Args:
            fmap: feature map to pool form
            proposal_boxes_batch_idx: proposal boxes with batch index inserted
                at the first channel
                (batch_idx, x1, y1, x2, y2, (z1, z2))[R, dim * 2 + 1]
            spatial_scale: the ratio of the size of the feature map and the
                original image (always <= 1)

        Returns:
            Tensor: pooled features from feature map [R, C, output_size]
        """
        raise NotImplementedError

    @abstractmethod
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
            binary_masks: binary segmentation masks [C, sdims]; C=number of instances
            proposal_boxes: proposal boxes
                (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            matched_gt_idx: index of matched ground truth box. The n-th
                box needs to correspond to the n-th channel inside the
                binary segmentation mask

        Returns:
            Tensor: pooled masks [N, output_size]
        """
        raise NotImplementedError
