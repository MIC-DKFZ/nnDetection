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
