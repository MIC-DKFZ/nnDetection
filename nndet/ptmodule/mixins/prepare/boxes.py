# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import List

from nndet.io.transforms import AbstractTransform, Instances2Boxes
from nndet.ptmodule.mixins.prepare.base import PrepareMixin


class BoxesPrepareMixin(PrepareMixin):
    def get_pre_transforms(self, plan: dict) -> List[AbstractTransform]:
        """
        Convert numbered instance mask to bounding boxes and classes.
        Requires input keys `target`, `instance_mapping` and `present_instances`.
        Results are saved into `target_boxes` and `target_classes`.

        Returns:
            List[AbstractTransform]: return a list of transformations

        Notes:
            make sure to call the super classes here!

        See Also:
            `Instances2Boxes`
        """
        transforms = super().get_pre_transforms(plan=plan)
        if not self.use_box_io():
            transforms.append(
                Instances2Boxes(
                    instance_key="target",
                    map_key="instance_mapping",
                    box_key="target_boxes",
                    class_key="target_classes",
                    present_instances="present_instances",
                )
            )
        return transforms
