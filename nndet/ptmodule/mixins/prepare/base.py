# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import List

from nndet.io.transforms import AbstractTransform, FindInstances


class PrepareMixin(ABC):
    @classmethod
    def use_box_io(cls) -> bool:
        """
        Returns:
            bool: `True` if boxes Dataloader and Augmentation should be
                used for this module. `False` otherwise.
        """
        return False

    def get_pre_transforms(self, plan: dict) -> List[AbstractTransform]:
        """
        Perform a sequence of transformations to the intput before passing it
        to the network. These transforamtions need to support pytorch tensors
        and are executed on the GPU.

        Returns:
            List[AbstractTransform]: return a list of transformations

        Notes:
            make sure to call the super classes here!
        """
        if self.use_box_io():
            transforms = []
        else:
            transforms = [
                FindInstances(
                    instance_key="target",
                    save_key="present_instances",
                ),
            ]
        return transforms
