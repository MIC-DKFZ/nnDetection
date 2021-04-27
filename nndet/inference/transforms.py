"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from loguru import logger
from typing import List, Tuple, Sequence

from nndet.io.transforms import Mirror, NoOp

from nndet.io.transforms.base import AbstractTransform


def get_tta_transforms(num_tta_transforms: int, seg: bool = True) -> Tuple[
        List[AbstractTransform], List[AbstractTransform]]:
    """
    Get tta transformations

    Args:
        num_tta_transforms: number of tta transformations; 0: no tta, 4: augments
            all directions in 2D, 8: augments all directions in 3D

    Returns:
        List[AbstractTransform]: transforms for TTA
        List[AbstractTransform]: inverted transformations for TTA
    """
    transforms = [NoOp()]
    inverse_transforms = [NoOp()]
    mirror_keys = ["data"]
    pred_mirror_keys = ["pred_seg"] if seg else ["pred_seg"]
    boxes_mirror_keys = ["pred_boxes"]

    if num_tta_transforms >= 4:
        logger.info("Adding 2D Mirror TTA for prediction.")
        transforms.append(Mirror(keys=mirror_keys, dims=(0,)))
        transforms.append(Mirror(keys=mirror_keys, dims=(1,)))
        transforms.append(Mirror(keys=mirror_keys, dims=(0, 1)))

        inverse_transforms.append(Mirror(keys=pred_mirror_keys,
                                         box_keys=boxes_mirror_keys, dims=(0,)))
        inverse_transforms.append(Mirror(keys=pred_mirror_keys,
                                         box_keys=boxes_mirror_keys, dims=(1,)))
        inverse_transforms.append(Mirror(keys=pred_mirror_keys,
                                         box_keys=boxes_mirror_keys, dims=(0, 1)))

    if num_tta_transforms >= 8:
        logger.info("Adding 3D Mirror TTA for prediction.")
        transforms.append(Mirror(keys=mirror_keys, dims=(2,)))
        transforms.append(Mirror(keys=mirror_keys, dims=(0, 2)))
        transforms.append(Mirror(keys=mirror_keys, dims=(1, 2)))
        transforms.append(Mirror(keys=mirror_keys, dims=(0, 1, 2)))

        inverse_transforms.append(Mirror(keys=pred_mirror_keys,
                                         box_keys=boxes_mirror_keys, dims=(2,)))
        inverse_transforms.append(Mirror(keys=pred_mirror_keys,
                                         box_keys=boxes_mirror_keys, dims=(0, 2)))
        inverse_transforms.append(Mirror(keys=pred_mirror_keys,
                                         box_keys=boxes_mirror_keys, dims=(1, 2)))
        inverse_transforms.append(Mirror(keys=pred_mirror_keys,
                                         box_keys=boxes_mirror_keys, dims=(0, 1, 2)))
    return transforms, inverse_transforms


class Inference2D(AbstractTransform):
    def __init__(self,
                 keys: Sequence[str],
                 ):
        """
        Helper transform to run inference with 2d models

        Args:
            keys: data keys to remove dimension from for inference
        """
        super().__init__(grad=False)
        self.keys = keys

    def forward(self, **data) -> dict:
        """
        Removes first spatial dimension (N, C, [removed], ax1, ax2)

        Args:
            **data: intput batch

        Returns:
            dict: transformed batch
        """
        for key in self.keys:
            data[key] = data[key][:, :, 0]
        return data
