from typing import TypeVar, List

import torch

from torchvision.ops.roi_align import roi_align


class Pooler():
    """
    Perform RoI Pooling for multi scale features
    """
    def __init__(self) -> None:
        pass

    def forward(self,
                features: List[torch.Tensor],
                porposals: torch.Tensor,
                ):
        pass

    def _pool_features(self,
                       features: torch.Tensor,
                       porposals: torch.Tensor,
                       ):
        pass

    def _find_pyramid_level(self,
                            porposals: torch.Tensor,
                            ):
        pass


class RoIAlign(Pooler):
    pass


PoolerType = TypeVar('PoolerType', bound=Pooler)
