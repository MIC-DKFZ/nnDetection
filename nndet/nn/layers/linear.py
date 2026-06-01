# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional, Type, Union

import torch
import torch.nn as nn

from nndet.nn.layers.wrapper import nd_act, nd_norm


class BaseNormLinearActDrop(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: Optional[Union[Callable[..., Type[nn.Module]], str]],
        act: Optional[Union[Callable[..., Type[nn.Module]], str]],
        dropout_rate: float = 0.0,
        bias: bool = None,
        norm_kwargs: Optional[dict] = None,
        act_inplace: Optional[bool] = None,
        act_kwargs: Optional[dict] = None,
        initializer: Callable[[nn.Module], None] = None,
    ):
        """
        Baseclass for ordering: norm -> linear -> activation -> dropout

        Args:
            in_channels: input channels
            out_channels: output_channels
            norm: type of normalization. If None, no normalization will be
                applied
            act: class of non linearity; if None no actication is used.
            dropout_rate: probability for dropout (i.e. probability to zero
                one element)
            bias: whether to include bias or not If None, the bias will be
                determined dynamicaly: False; if a normalization follows
                otherwise True;
            norm_kwargs: keyword arguments for normalization layer
            act_inplace: whether to perform activation inplce or not; If None,
                inplace will be determined dynamicaly: True; if a normalization
                follows otherwise False;
            act_kwargs: keyword arguments for non linearity layer.
            initializer: initilize weights
        """
        super().__init__()
        # process optional arguments
        norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        act_kwargs = {} if act_kwargs is None else act_kwargs

        if "inplace" in act_kwargs:
            raise ValueError("Use keyword argument to en-/disable inplace activations")
        act_inplace = False if act_inplace is None else act_inplace
        act_kwargs["inplace"] = act_inplace

        # process dynamic values
        bias = True if bias is None else bias

        if norm is not None:
            if isinstance(norm, str):
                _norm = nd_norm(norm, 1, out_channels, **norm_kwargs)
            else:
                _norm = norm(1, out_channels, **norm_kwargs)
            self.add_module("norm", _norm)

        fc = torch.nn.Linear(
            in_features=in_channels,
            out_features=out_channels,
            bias=bias,
        )
        self.add_module("fc", fc)

        if act is not None:
            if isinstance(act, str):
                _act = nd_act(act, 1, **act_kwargs)
            else:
                _act = act(**act_kwargs)
            self.add_module("act", _act)

        if dropout_rate > 0:
            self.add_module("dropout", torch.nn.Dropout(dropout_rate))

        if initializer is not None:
            self.apply(initializer)


class LayerLinearReluDrop(BaseNormLinearActDrop):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        add_norm: bool = True,
        add_act: bool = True,
        dropout_rate: float = 0.0,
        bias: bool = None,
        act_inplace: Optional[bool] = None,
        norm_eps: float = 1e-5,
        norm_elementwise_affine: bool = True,
        initializer: Callable[[nn.Module], None] = None,
    ):
        """
        Baseclass for ordering: linear -> norm -> activation -> dropout

        Args:
            in_channels: input channels
            out_channels: output_channels
            add_norm: add normalisation layer to block
            add_act: add activation layer to block
            dropout_rate: probability for dropout (i.e. probability to zero
                one element)
            bias: whether to include bias or not If None, the bias will be
                determined dynamicaly: False; if a normalization follows
                otherwise True;
            act_inplace: whether to perform activation inplce or not; If None,
                inplace will be determined dynamicaly: True; if a normalization
                follows otherwise False;
            norm_eps: value added to denominator of LayerNorm
            norm_elementwise_affine: add per element lernable affine paramters
                to layer norm
            initializer: initilize weights
        """
        norm = "Layer" if add_norm else None
        act = "ReLU" if add_act else None

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
            act=act,
            norm=norm,
            norm_kwargs={
                "eps": norm_eps,
                "elementwise_affine": norm_elementwise_affine,
            },
            act_inplace=act_inplace,
            initializer=initializer,
            dropout_rate=dropout_rate,
        )


class LayerLinearLReluDrop(BaseNormLinearActDrop):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        add_norm: bool = True,
        add_act: bool = True,
        dropout_rate: float = 0.0,
        bias: bool = None,
        act_inplace: Optional[bool] = None,
        act_negative_slope: float = 1e-2,
        norm_eps: float = 1e-5,
        norm_elementwise_affine: bool = True,
        initializer: Callable[[nn.Module], None] = None,
    ):
        """
        Baseclass for ordering: linear -> norm -> activation -> dropout

        Args:
            in_channels: input channels
            out_channels: output_channels
            add_norm: add normalisation layer to block
            add_act: add activation layer to block
            dropout_rate: probability for dropout (i.e. probability to zero
                one element)
            bias: whether to include bias or not If None, the bias will be
                determined dynamicaly: False; if a normalization follows
                otherwise True;
            act_inplace: whether to perform activation inplce or not; If None,
                inplace will be determined dynamicaly: True; if a normalization
                follows otherwise False;
            act_negative_slope: negative slope for LRelu activation
            norm_eps: value added to denominator of LayerNorm
            norm_elementwise_affine: add per element lernable affine paramters
                to layer norm
            initializer: initilize weights
        """
        norm = "Layer" if add_norm else None
        act = "LeakyReLU" if add_act else None

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
            act=act,
            act_kwargs={"negative_slope": act_negative_slope},
            norm=norm,
            norm_kwargs={
                "eps": norm_eps,
                "elementwise_affine": norm_elementwise_affine,
            },
            act_inplace=act_inplace,
            initializer=initializer,
            dropout_rate=dropout_rate,
        )
