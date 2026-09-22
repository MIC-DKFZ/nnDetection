# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Sequence, Type

from loguru import logger

from nndet.nn.backbone.abstract_ResEnc import WrapperResEncAbstractBackbone
from nndet.ptmodule.mixins.model.single import SingleStageMixin


class SingleStageMixin_ResEnc(SingleStageMixin):
    """
    :class:`SingleStageMixin` for backbones that build their own conv
    generator internally (from string-typed config) instead of receiving
    one from the mixin -- currently only ``_build_backbone`` needs to
    differ from the base implementation.
    """

    backbone_cls: Type[WrapperResEncAbstractBackbone] = ...  #: define class for backbone

    @classmethod
    def _build_backbone(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        patch_size: Optional[Sequence[int]] = None,
    ) -> WrapperResEncAbstractBackbone:
        """
        Build backbone network

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings
            patch_size: optionally provide the patch size
                to check compatibility with backbone

        Returns:
            AbstractBackbone: backbone instance
        """
        backbone: WrapperResEncAbstractBackbone = cls.backbone_cls.from_config_plan(
            backbone_cfg=model_cfg["backbone_kwargs"],
            plan_arch=plan_arch,
        )

        if patch_size is not None:
            if not backbone.check_patch_size(patch_size):
                raise ValueError(
                    f"Backbone {cls.backbone_cls.__name__} with absolute "
                    f"strides {backbone.get_absolute_strides()} is not compatible "
                    f"with patch size {patch_size}"
                )
            else:
                logger.info("Patch size check complete, backbone is compatible.")
        return backbone
