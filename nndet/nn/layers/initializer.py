# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from torch import nn

from nndet.utils.collections import CONV_TYPES, NORM_TYPES


class InitHe(object):
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
        self.mode = mode
        self.nonlinearity = nonlinearity

    def __call__(self, module: nn.Module):
        """
        Apply weight init

        Args:
            module: module to initialize weights of (only inits wights of convs)
        """
        if isinstance(module, CONV_TYPES):
            module.weight = nn.init.kaiming_normal_(
                module.weight,
                a=self.neg_slope,
                mode=self.mode,
                nonlinearity=self.nonlinearity,
            )
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class InitHeV2(object):
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
        self.mode = mode
        self.nonlinearity = nonlinearity

    def __call__(self, module: nn.Module):
        """
        Apply weight init

        Args:
            module: module to initialize weights of (only inits wights of convs)
        """
        if isinstance(module, CONV_TYPES):
            module.weight = nn.init.kaiming_normal_(
                module.weight,
                a=self.neg_slope,
                mode=self.mode,
                nonlinearity=self.nonlinearity,
            )
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
        elif isinstance(module, NORM_TYPES):
            if module.weight is not None:
                nn.init.constant_(module.weight, 1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
