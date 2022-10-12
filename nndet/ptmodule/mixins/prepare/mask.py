# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import List

from nndet.io.transforms import AbstractTransform
from nndet.io.transforms.instances import Instances2BinaryMasks
from nndet.ptmodule.mixins.prepare.base import PrepareMixin


class BinaryMasksPrepareMixin(PrepareMixin):
    def get_pre_transforms(self, plan: dict) -> List[AbstractTransform]:
        """
        Convert numbered instance mask to binary segmentation masks.
        Requires input keys `target` and `present_instances`.

        Returns:
            List[AbstractTransform]: return a list of transformations

        Notes:
            make sure to call the super classes here!

        See Also:
            `Instances2BinaryMasks`
        """
        transforms = super().get_pre_transforms(plan=plan)
        transforms.append(
            Instances2BinaryMasks(
                instance_key="target",
                binary_mask_key="target_binary_masks",
                present_instances="present_instances",
            )
        )
        return transforms
