import torch

from nndet.arch.heads.abstract import Classifier, CONV_TYPES
from nndet.arch.conv import nd_pool

from nndet.losses.classification import (
    BCEWithLogitsLossOneHot,
)


class RoIClassifierTwoMLP(Classifier):
    def __init__(self,
                 conv,
                 in_channels: int,
                 internal_channels: int,
                 num_classes: int,
                 ):
        super().__init__()
        self.dim = conv.dim

        self.fc = torch.nn.Sequential(
            *[
                torch.nn.Linear(
                    in_channels,
                    internal_channels,
                    ),
                torch.nn.Linear(
                    internal_channels,
                    num_classes,
                    ),
            ]
        )
        self.loss = BCEWithLogitsLossOneHot(
            num_classes=num_classes,
            reduction="sum",
        )

    def forward(self, features):
        return self.fc(features.view(features.shape[0], -1))

    # TODO: CE refactor
    def compute_loss(self,
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
        return self.loss(pred_logits, targets)

    def box_logits_to_probs(self,
                            box_logits: torch.Tensor,
                            ) -> torch.Tensor:
        """
        Convert bounding box logits to probabilities

        Args:
            box_logits (Tensor): bounding box logits
                [N, C], C=number of classes

        Returns:
            Tensor: probabilities; [N, C], C=number of classes
        """
        return torch.sigmoid(box_logits)
