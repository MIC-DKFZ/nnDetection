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

from typing import Optional

import torch
from torch import nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        feedforward_dim: int,
        num_layers: int,
        activation: nn.Module,
        output_dim: Optional[int] = None,
        ffn_drop: Optional[float] = 0.0,
        fc_bias: Optional[bool] = True,
    ):
        """
        The implementation of a fully connected network with identity connection.
        (lin -> act -> dropout) X (num_layers - 1) -> (lin -> dropout)

        Args:
            embed_dim: The feature dimension.
            feedforward_dim: The hidden dimension of FFNs.
            num_layers: The number of fully-connected layers in FFNs.
            activation: The activation layer used in FFNs.
            output_dim: The output feature dimension of FFNs. If None, the
                `embed_dim` will be used.
            ffn_drop: Probability of an element to be zeroed in FFN.
            fc_bias: whether to use bias in linear layers
        """
        super().__init__()
        assert num_layers >= 2, "num_layers should be no less " f"than 2. got {num_layers}."
        self.embed_dim = embed_dim
        self.feedforward_dim = feedforward_dim
        self.num_fcs = num_layers
        self.activation = activation

        output_dim = embed_dim if output_dim is None else output_dim

        layers = []
        in_channels = embed_dim
        for _ in range(num_layers - 1):
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

    def forward(
        self,
        x: torch.Tensor,
        identity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        if identity is not None:
            out = out + identity
        return out


class ReluMLP(MLP):
    def __init__(
        self,
        embed_dim: int,
        feedforward_dim: int,
        output_dim: int,
        num_layers: int,
    ):
        """
        Very simple multi-layer perceptron with a relu activation. Without
        dropout or residual connections. Requires output_dim explicitly.

        Args:
            embed_dim: number of input neurons
            feedforward_dim: number of neurons in the hidden layers
            output_dim: number of output neurons
            num_layers: number of layers
        """
        super().__init__(
            embed_dim=embed_dim,
            feedforward_dim=feedforward_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            ffn_drop=0.0,
            activation=nn.ReLU(inplace=True),
            fc_bias=True,
        )


class ReluDropIdentityMLP(MLP):
    def __init__(
        self,
        embed_dim: int,
        feedforward_dim: int,
        num_layers: int,
        output_dim: Optional[int] = None,
        ffn_drop: Optional[float] = 0.0,
        fc_bias: Optional[bool] = True,
    ):
        """
        MLP with ReLU activation, dropout and a skip connection. Used in
        transformer layers.

        Args:
            embed_dim: number of input neurons
            feedforward_dim: number of neurons in the hidden layers
            output_dim: number of output neurons
            num_layers: number of layers
        """
        super().__init__(
            embed_dim=embed_dim,
            feedforward_dim=feedforward_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            activation=nn.ReLU(inplace=True),
            ffn_drop=ffn_drop,
            fc_bias=fc_bias,
        )
