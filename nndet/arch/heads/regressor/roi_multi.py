# """
# Contains RoI Heads which regress each anchor for all clases
# """
# from typing import Optional
# import torch

# from nndet.losses import SmoothL1Loss
# from nndet.arch.heads.regressor2.roi.base import BaseRoIRegressor


# class RoIRegressorFC(BaseRoIRegressor):
#     def __init__(self,
#                  conv,
#                  in_channels: int,
#                  internal_channels: int,
#                  num_classes: int,
#                  beta: float = 1.,
#                  reduction: Optional[str] = "sum",
#                  loss_weight: float = 1.,
#                  ):
#         """
#         Base class for RoI regression heads

#         Args:
#             conv: conv generator
#             in_channels: number of input channels
#             internal_channels: number of internal channels
#             num_classes: number of classes
#             beta: L1 to L2 change point.
#                 For beta values < 1e-5, L1 loss is computed.
#             reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
#             loss_weight: scalar to balance multiple losses
#         """
#         super().__init__(
#             conv=conv,
#             in_channels=in_channels,
#             internal_channels=internal_channels,
#             num_classes=num_classes,
#         )
#         self.fc = torch.nn.Linear(
#             in_features=self.in_channels,
#             out_features=self.num_classes * self.dim * 2,
#             bias=True,
#         )
#         self.loss = SmoothL1Loss(
#             beta=beta,
#             reduction=reduction,
#             loss_weight=loss_weight,
#             )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Compute box deltas per class

#         Args:
#             x: input tensor [N, in_channels]

#         Returns:
#             torch.Tensor: [N, num_classes, dim * 2]
#         """
#         x = self.fc(x.view(x.shape[0], -1))
#         return x.view(x.shape[0], self.num_classes, self.dim * 2)

#     def compute_loss(self,
#                      pred_deltas: torch.Tensor,
#                      target_deltas: torch.Tensor,
#                      target_labels: torch.Tensor,
#                      **kwargs,
#                      ) -> torch.Tensor:
#         """
#         Compute regression loss

#         Args:
#             pred_deltas: predicted bounding box deltas
#                 [N, (num_classes), dim * 2]
#             target_deltas: target bounding box deltas [N,  dim * 2]
#             target_labels: target labels [N]

#         Returns:
#             Tensor: computed loss
#         """
#         return self.loss(pred_deltas[:, target_labels],
#                          target_deltas,
#                          **kwargs,
#                          )

# from abc import abstractmethod
# from typing import Optional

# import torch


# class RoIRegressor(torch.nn.Module):
#     def __init__(self,
#                  conv,
#                  in_channels: int,
#                  internal_channels: int,
#                  num_classes: int,
#                  ):
#         """
#         Base class for RoI regression heads

#         Args:
#             conv: conv generator
#             in_channels: number of input channels
#             internal_channels: number of internal channels
#             num_classes: number of classes
#         """
#         super().__init__()
#         self.dim = conv.dim
#         self.in_channels = in_channels
#         self.internal_channels = internal_channels
#         self.num_classes = num_classes

#     @abstractmethod
#     def compute_loss(self,
#                      pred_deltas: torch.Tensor,
#                      target_deltas: torch.Tensor,
#                      target_labels: torch.Tensor,
#                      **kwargs,
#                      ) -> torch.Tensor:
#         """
#         Compute regression loss

#         Args:
#             pred_deltas: predicted bounding box deltas
#                 [N, (num_classes), dim * 2]
#             target_deltas: target bounding box deltas [N,  dim * 2]
#             target_labels: target labels [N]

#         Returns:
#             Tensor: computed loss
#         """
#         raise NotImplementedError


# class BaseRoIRegressor(RoIRegressor):
#     def __init__(self,
#                  conv,
#                  in_channels: int,
#                  internal_channels: int,
#                  num_classes: int,
#                  ):
#         """
#         Base class for RoI regression heads

#         Args:
#             conv: conv generator
#             in_channels: number of input channels
#             internal_channels: number of internal channels
#             num_classes: number of classes
#         """
#         super().__init__(
#             conv=conv,
#             in_channels=in_channels,
#             internal_channels=internal_channels,
#             num_classes=num_classes,
#         )
#         self.loss: Optional[torch.nn.Module] = None

#     def compute_loss(self,
#                     pred_deltas: torch.Tensor,
#                     target_deltas: torch.Tensor,
#                     target_labels: torch.Tensor,
#                     **kwargs,
#                     ) -> torch.Tensor:
#         """
#         Compute regression loss

#         Args:
#             pred_deltas: predicted bounding box deltas
#                 [N, (num_classes), dim * 2]
#             target_deltas: target bounding box deltas [N,  dim * 2]
#             target_labels: target labels [N]

#         Returns:
#             Tensor: computed loss
#         """
#         return self.loss(pred_deltas[:, target_labels], target_deltas, **kwargs)
