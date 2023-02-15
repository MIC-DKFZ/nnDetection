# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from detrex licensed under
# SPDX-FileCopyrightText: 2022, The IDEA Authors
# SPDX-License-Identifier: Apache-2.0

# Parts of this code are from detr licensed under
# SPDX-FileCopyrightText: 2020, Facebook, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Misc functions, including distributed helpers.
Mostly copy-paste from torchvision references.
"""
from typing import Optional

import torch
from torch import nn as nn
from torch.nn import functional as F


class SimpleFCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
    ):
        """
        Very simple multi-layer perceptron with a relu activation. Without
        dropout or extra residual connections.

        Args:
            input_dim: number of input neurons
            hidden_dim: number of neurons in the hidden layers
            output_dim: number of output neurons
            num_layers: number of layers
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call of the fully connected network

        Args:
            x: input tensor

        Returns:

        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class FCN(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        feedforward_dim: int = 1024,
        output_dim: int = None,
        num_fcs: int = 2,
        activation: nn.Module = nn.ReLU(inplace=True),
        ffn_drop: Optional[float] = 0.0,
        fc_bias: Optional[bool] = True,
        add_identity: Optional[bool] = True,
    ):
        """
        The implementation of a fully connected network with identity connection.

        Args:
            embed_dim: The feature dimension.
            feedforward_dim: The hidden dimension of FFNs.
            output_dim: The output feature dimension of FFNs. If None, the
                `embed_dim` will be used.
            num_fcs: The number of fully-connected layers in FFNs.
            activation: The activation layer used in FFNs.
            ffn_drop: Probability of an element to be zeroed in FFN.
            add_identity: Whether to add the identity connection.
        """
        super(FCN, self).__init__()
        assert num_fcs >= 2, "num_fcs should be no less " f"than 2. got {num_fcs}."
        self.embed_dim = embed_dim
        self.feedforward_dim = feedforward_dim
        self.num_fcs = num_fcs
        self.activation = activation

        output_dim = embed_dim if output_dim is None else output_dim

        layers = []
        in_channels = embed_dim
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_dim, bias=fc_bias),
                    self.activation,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_dim
        layers.append(nn.Linear(feedforward_dim, output_dim, bias=fc_bias))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity

    def forward(self, x: torch.Tensor, identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward function of `FFN`.

        Args:
            x: the input tensor used in `FFN` layers.
            identity: the tensor with the same shape as `x`, which will be used
                for identity addition. If None, `x` will be used.

        Returns:
            the forward results of `FFN` layer
        """
        out = self.layers(x)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out
