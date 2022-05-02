from abc import abstractmethod
from typing import List, Optional, Tuple, TypeVar, Union

import torch
from torch import Tensor

from nndet.core.boxes.ops import box_size, expand_to_boxes
from nndet.core.rois.roi_align import roi_align

NDSIZE = Union[Tuple[int, int], Tuple[int, int, int]]


# TODO: docs with feature output size instead of simple output size
class RoIPooler(torch.nn.Module):
    def __init__(
        self,
        feature_output_size: NDSIZE,
        mask_output_size: Optional[NDSIZE] = None,
    ):
        """
        Perform RoI Pooling for multi scale features
        """
        super().__init__()
        self.feature_output_size = feature_output_size
        self.mask_output_size = mask_output_size

    def forward(
        self,
        features: List[torch.Tensor],
        proposal_boxes: torch.Tensor,
        batch_idx: torch.Tensor,
        image_size: NDSIZE,
    ) -> torch.Tensor:
        """
        Perform multiscale pyramid pooling

        Args:
            features: feature maps which should be used for pooling
                from backbone/fpn
            proposal_boxes: box proposals
                (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            batch_idx: original batch index of each proposal [N]
            image_size: spatial size of image

        Returns:
            Tensor: extracted features [N, C, output_size]
        """
        # TODO: IMPORTANT!!!!!!!!!
        # THIS ONLY WORKS FOR ISOTROPIC POOLING
        # NEED TO GENERALIZE TO NON ISOTROPIC POOLING
        image_size_tensor = torch.tensor(
            image_size,
            dtype=proposal_boxes.dtype,
            device=proposal_boxes.device,
        )
        # normalize boes to [0, 1]
        proposal_boxes_norm = proposal_boxes / expand_to_boxes(image_size_tensor)

        # TODO: dynamically infer scale, these normlizations are wrong
        proprosals_prepared = torch.cat(
            [batch_idx[:, None], proposal_boxes],
            dim=1,
        )

        if len(features) == 1:
            spatial_scale = tuple(
                features[0].shape[i + 2] / image_size_tensor[i]
                for i in range(len(image_size_tensor))
            )
            output = self._pool_features(
                fmap=features[0],
                proposals=proprosals_prepared,
                spatial_scale=spatial_scale,
            )
        else:  # determine level dynamically
            proposal_levels = self._find_pyramid_level(
                proposal_boxes_norm=proposal_boxes_norm,
                features=features,
                image_size=image_size,
            )

            # TODO: need to check dtype due to autocast stuff
            output = torch.zeros(
                [
                    proposal_boxes_norm.shape[0],
                    features[0].shape[1],
                    *self.feature_output_size,
                ],
                dtype=features[0].dtype,
                device=features[0].device,
            )

            for idx, fmap in enumerate(features):
                spatial_scale = tuple(
                    features[0].shape[i + 2] / image_size_tensor[i]
                    for i in range(len(image_size_tensor))
                )
                idx = torch.where(proposal_levels == idx)[0]
                if idx.numel() > 0:
                    output[idx] = self._pool_features(
                        fmap=fmap,
                        proposals=proprosals_prepared[idx],
                        spatial_scale=spatial_scale,
                    )
        return output

    @abstractmethod
    @torch.no_grad()
    def _find_pyramid_level(
        self,
        proposal_boxes_norm: torch.Tensor,
        features: List[torch.Tensor],
        image_size: Union[Tuple[int, int], Tuple[int, int, int]],
    ) -> torch.Tensor:
        """
        Assign proposals to pyramid levels for pooling

        Args:
            proposal_boxes_norm: batch expanded proposals for pooling
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
        proposals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pooling feature for proposals from given feature map

        Args:
            fmap: feature map to pool form
            proposals: proposals to extract feature for
                (batch_idx, x1, y1, x2, y2, (z1, z2))[R, dim * 2]

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
            masks: binary segmentation masks [C, sdims]; C=number of instances
            proposal_boxes: proposal boxes to pool
                (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            matched_gt_idx: index of matched ground truth box. The n-th
                box needs to correspond to the n-th channel inside the
                binary segmentation mask

        Returns:
            Tensor: pooled masks [N, output_size]
        """
        raise NotImplementedError


class RoIAlignNaiveAssign(RoIPooler):
    @torch.no_grad()
    def _find_pyramid_level(
        self,
        proposal_boxes_norm: torch.Tensor,
        features: List[torch.Tensor],
        image_size: Union[Tuple[int, int], Tuple[int, int, int]],
    ) -> torch.Tensor:
        """
        Assign proposals to pyramid levels for pooling
        Proposals with an image size of
        """
        # norm proposals. Proposals with 3/4 of the image size will be mapped to 1
        # We normalize the box size instead of the area/vol
        # since this should give better numerical results especially
        # when using mixed precision (i.e. 128^3 does not fit float16)
        normed_size = box_size(proposal_boxes_norm) * 1.33  # [N, 3]

        if len(image_size) == 2:
            v = torch.log2((normed_size[:, 0] * normed_size[:, 1]).sqrt())
        elif len(image_size) == 3:
            v = torch.log2(
                (normed_size[:, 0] * normed_size[:, 1] * normed_size[:, 2]) ** (1 / 3)
            )
        else:
            raise ValueError(f"Image size needs to be 2D or 3d, received {image_size}.")

        level = torch.floor(v * len(features)) + len(features)
        return level.clamp_(min=0, max=len(features)).to(dtype=torch.int)

    def _pool_features(
        self,
        fmap: torch.Tensor,
        proposals: torch.Tensor,
        spatial_scale: Union[float, Tuple[float]],
    ) -> torch.Tensor:
        """
        Pooling feature for proposals from given feature map
        """
        return roi_align(
            input=fmap,
            boxes=proposals,
            output_size=self.feature_output_size,
            spatial_scale=spatial_scale,
            aligned=True,
            sampling_ratio=2,
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
            p_boxes_prepared = torch.cat([m_idx[:, None], p_boxes], dim=1)
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
                        boxes=p_boxes_prepared,
                        output_size=output_size,
                        spatial_scale=1.0,
                        aligned=True,
                    )[:, 0]
                )
        return pooled_masks


RoIPoolerType = TypeVar("RoIPoolerType", bound=RoIPooler)
