# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

import pydoc
import torch
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from loguru import logger

from nndet.nn.backbone.abstract_ResEnc import ResEncAbstract, WrapperResEncAbstractBackbone
from nndet.utils.make_json_safe_value import to_python


class ResEnc(ResEncAbstract):
    def __init__(self,
                 config) -> None:

        input_channels = config['plan_arch']['in_channels']

        input_shape = config['backbone_cfg']['patch_size']
        n_stages=config['backbone_cfg']['n_stages']
        features_per_stage=config['backbone_cfg']['features_per_stage']
        conv_op=pydoc.locate(config['backbone_cfg']['conv_op'])
        kernel_sizes=config['backbone_cfg']['kernel_sizes']
        strides=config['backbone_cfg']['strides']
        n_blocks_per_stage=config['backbone_cfg']['n_blocks_per_stage']
        conv_bias=config['backbone_cfg']['conv_bias']
        norm_op=pydoc.locate(config['backbone_cfg']['norm_op'])
        norm_op_kwargs=config['backbone_cfg']['norm_op_kwargs']
        dropout_op=config['backbone_cfg']['dropout_op']
        dropout_op_kwargs=config['backbone_cfg']['dropout_op_kwargs']
        nonlin=pydoc.locate(config['backbone_cfg']['nonlin'])
        nonlin_kwargs=config['backbone_cfg']['nonlin_kwargs']
        return_skips=config['backbone_cfg']['return_skips']

        assert input_shape is not None
        assert len(input_shape) == 3

        super().__init__()

        self.config=config

        self.input_shape = input_shape

        self.input_channels = input_channels

        # Encoder using ResidualEncoder
        self.resenc = ResidualEncoder(input_channels=input_channels, n_stages=n_stages, features_per_stage=features_per_stage,
                 conv_op=conv_op,kernel_sizes=kernel_sizes,
                 strides=strides,n_blocks_per_stage=n_blocks_per_stage,conv_bias=conv_bias,norm_op=norm_op,
                 norm_op_kwargs=norm_op_kwargs,
                 dropout_op=dropout_op,
                 dropout_op_kwargs=dropout_op_kwargs,
                 nonlin=nonlin,
                 nonlin_kwargs=nonlin_kwargs,
                 return_skips=return_skips
        )

    def forward(self, x):
        if self.resenc.stem is not None:
            x = self.resenc.stem(x)
        ret = []
        for s in self.resenc.stages:
            x = s(x)
            ret.append(x)
        if self.resenc.return_skips:
            return ret
        else:
            return ret[-1]

    def get_channels(self):
        # Return the specific channel configuration for this architecture
        self.out_channels=self.resenc.output_channels
        return self.out_channels

    def get_relative_strides(self):
        # Return the strides for this architecture
        return self.resenc.strides

class ResidualEncoderUNetbackBoneWrapper(WrapperResEncAbstractBackbone):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        """
        Residual Backbone
        """
        self.backbone=self.from_config_plan(config['backbone_kwargs'],config['plan_arch'])

    @classmethod
    def from_config_plan(
        cls,
        backbone_cfg: dict,
        plan_arch: dict,
    ):
        """
        Instantiate Backbone from given configs.

        Args
            backbone_cfg: backbone configuration
        """
        logger.info(f"Building:: backbone {cls.__name__}: {backbone_cfg} ")
        logger.info(f"Building:: Arch {cls.__name__} (content mainly not used): {plan_arch} ")
        # parse config and plan
        backbone_cfg=to_python(backbone_cfg)
        plan_arch=to_python(plan_arch)
        #for now i fix everything
        config={'backbone_cfg': backbone_cfg, 'plan_arch': plan_arch}
        backbone = ResEnc(config=config)
        return backbone


class ResEnc_dyn(ResEncAbstract):
    def __init__(self,
                 config) -> None:


        plan_arch = config["plan_arch"]
        bb_cfg   = config['backbone_cfg']
        input_channels = plan_arch['in_channels']
        n_stages=len(plan_arch['conv_kernels'])
        blocks_per_stage=[1, 3, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
        n_blocks_per_stage=blocks_per_stage[:n_stages]

        start_channels=plan_arch['start_channels']
        max_channels=plan_arch['max_channels']
        conv_op=pydoc.locate(bb_cfg['conv_op'])
        kernel_sizes=plan_arch['conv_kernels']
        strides = [[1, 1, 1]] + [s for s in config["plan_arch"]["strides"]]

        conv_bias=bb_cfg['conv_bias']
        norm_op=pydoc.locate(bb_cfg['norm_op'])
        norm_op_kwargs=bb_cfg['norm_op_kwargs']
        dropout_op=bb_cfg['dropout_op']
        dropout_op_kwargs=bb_cfg['dropout_op_kwargs']
        nonlin=pydoc.locate(bb_cfg['nonlin'])
        nonlin_kwargs=bb_cfg['nonlin_kwargs']
        return_skips=bb_cfg['return_skips']

        num_levels = len(kernel_sizes)

        features_per_stage: List[int] = []

        # build levels
        for idx in range(num_levels):
            _out_channels = min(
            start_channels * (2**idx),
            max_channels,
        )
            features_per_stage.append(_out_channels)

        super().__init__()

        self.config=config

        self.input_channels = input_channels

        # Encoder using ResidualEncoder
        self.resenc = ResidualEncoder(input_channels=input_channels, n_stages=n_stages, features_per_stage=features_per_stage,
                 conv_op=conv_op,kernel_sizes=kernel_sizes,
                 strides=strides,n_blocks_per_stage=n_blocks_per_stage,conv_bias=conv_bias,norm_op=norm_op,
                 norm_op_kwargs=norm_op_kwargs,
                 dropout_op=dropout_op,
                 dropout_op_kwargs=dropout_op_kwargs,
                 nonlin=nonlin,
                 nonlin_kwargs=nonlin_kwargs,
                 return_skips=return_skips
        )

    def forward(self, x):
        if self.resenc.stem is not None:
            x = self.resenc.stem(x)
        ret = []
        for s in self.resenc.stages:
            x = s(x)
            ret.append(x)
        if self.resenc.return_skips:
            return ret
        else:
            return ret[-1]

    def get_channels(self):
        # Return the specific channel configuration for this architecture
        self.out_channels=self.resenc.output_channels
        return self.out_channels

    def get_relative_strides(self):
        # Return the strides for this architecture
        return self.resenc.strides


class ResidualEncoderUNetbackBoneWrapper_dyn(WrapperResEncAbstractBackbone):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        """
        Residual Backbone
        """
        self.backbone=self.from_config_plan(config['backbone_kwargs'],config['plan_arch'])

    @classmethod
    def from_config_plan(
        cls,
        backbone_cfg: dict,
        plan_arch: dict,
    ):
        """
        Instantiate Backbone from given configs.

        Args
            backbone_cfg: backbone configuration
        """
        logger.info(f"Building:: backbone {cls.__name__}: {backbone_cfg} ")
        logger.info(f"Building:: Arch {cls.__name__} (content mainly not used): {plan_arch} ")
        # parse config and plan
        backbone_cfg=to_python(backbone_cfg)
        plan_arch=to_python(plan_arch)
        #for now i fix everything
        config={'backbone_cfg': backbone_cfg, 'plan_arch': plan_arch}
        backbone = ResEnc_dyn(config=config)
        return backbone
