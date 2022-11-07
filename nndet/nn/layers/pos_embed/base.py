from abc import abstractmethod

import torch


class BasePositionEmbedding(torch.nn.Module):
    def __init__(self, in_channels: int) -> None:
        """
        Base class to implement positional embeddings

        Args:
            in_channels: number of input features
        """
        super().__init__()
        self.num_pos_feats = in_channels

    @abstractmethod
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Generate embedding

        Args:
            data: input feature map. [N, C, dims]
            N = batch size, C = number of channels,
            dims = spatial dimensions

        Returns:
            torch.Tensor: spatial embedding [] #TODO: insert here
        """
        raise NotImplementedError
