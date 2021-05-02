from abc import abstractmethod
from typing import TypeVar, List, Union, Tuple
from nndet.core import boxes

import torch

from torchvision.ops.roi_align import roi_align as _roi_align

from nndet.core.boxes.utils import box_size, expand_to_boxes, permute_boxes


class Pooler(torch.nn.Module):
    def __init__(self,
                 output_size: Union[Tuple[int, int], Tuple[int, int, int]],
                 ):
        """
        Perform RoI Pooling for multi scale features
        """
        self.output_szie = output_size

    def forward(self,
                proposals: torch.Tensor,
                features: List[torch.Tensor],
                image_size: Union[Tuple[int, int], Tuple[int, int, int]]
                ) -> torch.Tensor:
        """
        Perform multiscale pyramid pooling

        Args:
            proposals: batch expanded proposals for pooling
                (batch_idx, x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            features: feature maps which should be used for pooling
                from backbone/fpn
            image_size: spatial size of image

        Returns:
            Tensor: extracted features [N, C, output_size]
        """
        # TODO: IMPORTANT!!!!!!!!!
        # THIS ONLY WORKS FOR ISOTROPIC POOLING
        # NEED TO GENERALIZE TO NON ISOTROPIC POOLING
        image_size_tensor = torch.tensor(image_size,
                                    dtype=proposals.dtype,
                                    device=proposals.device,
                                    )
        # normalize boes to [0, 1]
        proposals[:, 1:] = proposals[:, 1:] / expand_to_boxes(image_size_tensor)

        proposal_levels = self._find_pyramid_level(
            proposals=proposals,
            features=features,
            image_size=image_size,
            )

        # TODO: need to check dtype due to autocast stuff
        output = torch.zeros(
            [proposals.shape[0], features[0].shape[1], *self.output_szie],
            dtype=features.dtype,
            device=features.device,
        )

        for idx, fmap in enumerate(features):
            idx = torch.where(proposal_levels == idx)[0]
            if idx:
                output[idx] = self._pool_features(
                    fmap=fmap,
                    proposals=proposals[idx],
                )
        return output

    @abstractmethod
    @torch.no_grad
    def _find_pyramid_level(self,
                            normed_proposals: torch.Tensor,
                            features: List[torch.Tensor],
                            image_size: Union[Tuple[int, int],
                                              Tuple[int, int, int]],
                            ) -> torch.Tensor:
        """
        Assign proposals to pyramid levels for pooling

        Args:
            proposals: batch expanded proposals for pooling
                (batch_idx, x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            features: feature maps which should be used for pooling
                from backbone/fpn
            image_size: spatial size of image

        Returns:
            Tensor: Index of pyramid level for each for pooling
        """
        raise NotImplementedError

    @abstractmethod
    def _pool_features(self,
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


class RoIAlignNaiveAssign(Pooler):
    @torch.no_grad
    def _find_pyramid_level(self,
                            normed_proposals: torch.Tensor,
                            features: List[torch.Tensor],
                            image_size: Union[Tuple[int, int],
                                              Tuple[int, int, int]],
                            ) -> torch.Tensor:
        """
        Assign proposals to pyramid levels for pooling
        Proposals with an image size of 
        """
        # norm proposals. Proposals with 3/4 of the image size will be mapped to 1
        # We normalize the box size instead of the area/vol
        # since this should give better numerical results especially
        # when using mixed precision (i.e. 128^3 does not fit float16)
        normed_size = box_size(normed_proposals[:, 1:]) * 1.33 # [N, 3]

        if len(image_size) == 2:
            v = torch.log2((normed_size[:, 1] * normed_size[:, 2]).sqrt())
        elif len(image_size) == 3:
            v = torch.log2((normed_size[:, 1] * normed_size[:, 2] * normed_size[:, 3]) ** (1 / 3))
        else:
            raise ValueError(f"Image size needs to be 2D or 3d, received {image_size}.")

        level = torch.floor(v * len(features)) + len(features)
        return level.clamp_(min=0, max=len(features)).to(dtype=torch.int)

    def _pool_features(self,
                       fmap: torch.Tensor,
                       proposals: torch.Tensor,
                       ) -> torch.Tensor:
        """
        Pooling feature for proposals from given feature map
        """
        # TODO: replace with own ROI Align and remove permute!
        # TODO: wirte own ROI Align with general scaling parameter?
        proposals[:, 1:] = permute_boxes(proposals[:, 1:], dims=[1, 0])
        return _roi_align(
            input=fmap,
            boxes=proposals,
            output_size=self.output_szie,
            aligned=True,
        )


PoolerType = TypeVar('PoolerType', bound=Pooler)
