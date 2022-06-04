# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from torch import nn


class InitWeights_He(object):
    def __init__(
        self,
        neg_slope: float = 1e-2,
        mode: str = "fan_in",
        nonlinearity="leaky_relu",
    ):
        """
        Init weights according to https://arxiv.org/abs/1502.01852

        Args:
            neg_slope (float, optional): the negative slope of the rectifier
                used after this layer (only with 'leaky_relu').
                Defaults to 1e-2.
            mode: mode of `kaiming_normal_` mode
            nonlinearity: name of non linear function. Recommended only with
                relu and leaky relu
        """
        self.neg_slope = neg_slope

    def __call__(self, module: nn.Module):
        """
        Apply weight init

        Args:
            module: module to initialize weights of (only inits wights of convs)
        """
        if isinstance(
            module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d, nn.ConvTranspose3d)
        ):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
