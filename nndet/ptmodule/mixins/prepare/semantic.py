# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import List

from nndet.io.transforms import AbstractTransform, Instances2Fg, Instances2Segmentation
from nndet.ptmodule.mixins.prepare.base import PrepareMixin


class SemanticPrepareMixin(PrepareMixin):
    def get_pre_transforms(self, plan: dict) -> List[AbstractTransform]:
        """
        Convert numbered instance mask to semantic segmentation.
        Requires input keys `target` and `present_instances`.

        Returns:
            List[AbstractTransform]: return a list of transformations

        Notes:
            make sure to call the super classes here!

        See Also:
            `Instances2Segmentation`
        """
        transforms = super().get_pre_transforms(plan=plan)
        transforms.append(
            Instances2Segmentation(
                instance_key="target",
                map_key="instance_mapping",
                seg_key="target_seg",
                present_instances="present_instances",
            )
        )
        return transforms


class SemanticFgPrepareMixin(PrepareMixin):
    def get_pre_transforms(self, plan: dict) -> List[AbstractTransform]:
        """
        Convert numbered instance mask to foreground segmentation.
        Requires input keys `target`.

        Returns:
            List[AbstractTransform]: return a list of transformations

        Notes:
            make sure to call the super classes here!

        See Also:
            `Instances2Fg`
        """
        trafos = super().get_pre_transforms(plan=plan)
        trafos.append(
            Instances2Fg(
                instance_key="target",
                seg_key="target_seg",
            )
        )
        return trafos
