import torch

from nndet.arch.conv import nd_pool
from nndet.arch.heads.abstract import Regressor


class RoIRegressorConv(Regressor):
    def __init__(
        self,
        conv,
        in_channels: int,
        internal_channels: int,
    ):
        super().__init__()
        self.dim = conv.dim

        self.conv_internal = torch.nn.Sequential(
            *[
                conv(
                    in_channels,
                    internal_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                conv(
                    internal_channels,
                    internal_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nd_pool("AdaptiveAvg", self.dim, 1),
            ]
        )
        self.fc = torch.nn.Linear(
            internal_channels,
            self.dim * 2,
        )

    def forward(self, features):
        x = self.conv_internal(features)  # N, C, spatial -> N, C, 1
        return self.fc(x.view(x.shape[0], -1))

    def compute_loss(
        self,
        pred_logits: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute classification loss (cross entropy loss)

        Args:
            pred_logits (Tensor): predicted logits
            targets (Tensor): classification targets

        Returns:
            Tensor: classification loss
        """
        return torch.nn.L1Loss(reduction="sum")(pred_logits, targets)
