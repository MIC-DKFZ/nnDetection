from abc import ABC
from typing import List

from nndet.io.transforms import (
    AbstractTransform,
    FindInstances,
    Instances2Boxes,
    Instances2Fg,
    Instances2Segmentation,
)


class PrepareMixin(ABC):
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
        transforms = [
            FindInstances(
                instance_key="target",
                save_key="present_instances",
            ),
        ]
        return transforms


class BoxPrepareMixin(PrepareMixin):
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


class InstancePrepareMixin(PrepareMixin):
    def get_pre_transforms(self, plan: dict) -> List[AbstractTransform]:
        """
        Convert numbered instance mask to binary segmentation masks.
        Requires input keys `target` and `present_instances`.

        Returns:
            List[AbstractTransform]: return a list of transformations

        Notes:
            make sure to call the super classes here!

        See Also:
            # TODO

        """
        raise NotImplementedError
        trafos = super().get_pre_transforms(plan=plan)
        return trafos
