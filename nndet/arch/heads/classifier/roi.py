from typing import Tuple, Union

import torch

from nndet.arch.conv import nd_pool
from nndet.arch.heads.abstract import Classifier


class RoIClassifierTwoMLP(Classifier):
    def __init__(
        self,
        conv,
        output_size: Union[Tuple[int, int], Tuple[int, int, int]],
        in_channels: int,
        internal_channels: int,
        num_classes: int,
    ):
        super().__init__()
        self.dim = conv.dim

        # self.fc = torch.nn.Sequential(
        #     *[
        #         torch.nn.Linear(
        #             in_channels * output_size[0] * output_size[1], # TODO: 2d only
        #             internal_channels,
        #         ),
        #         torch.nn.ReLU(),
        #         torch.nn.Linear(
        #             internal_channels,
        #             internal_channels,
        #         ),
        #         torch.nn.ReLU(),
        #         torch.nn.Linear(
        #             internal_channels,
        #             num_classes + 1,
        #         ),
        #     ]
        # )
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
            num_classes + 1,
        )
        self.loss = torch.nn.CrossEntropyLoss(
            reduction="sum",
        )

    def forward(self, features):
        x = self.conv_internal(features)  # N, C, spatial -> N, C, 1
        return self.fc(x.view(x.shape[0], -1))

        # return self.fc(features.view(features.shape[0], -1))

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
        return self.loss(pred_logits, targets)

    def box_logits_to_probs(
        self,
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
        return torch.nn.functional.softmax(box_logits, dim=1)[:, 1:]
