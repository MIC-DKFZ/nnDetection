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

        Raises:
            NotImplementedError: needs to be overwritten in the subclasses

        Returns:
            List[AbstractTransform]: return a list of transformations

        Notes:
            make sure to call the super classes here!
        """
        return []


class BoxPrepareMixin(PrepareMixin):
    def get_pre_transforms(self, plan: dict) -> List[AbstractTransform]:
        """
        Search for unqiue instances -> Instances to Boxes

        Returns:
            List[AbstractTransform]: return a list of transformations

        Notes:
            make sure to call the super classes here!
        """
        transforms = super().get_pre_transforms(plan=plan)
        transforms.append(
            FindInstances(
                instance_key="target",
                save_key="present_instances",
            )
        )
        transforms.append(
            Instances2Boxes(
                instance_key="target",
                map_key="instance_mapping",
                box_key="boxes",
                class_key="classes",
                present_instances="present_instances",
            )
        )
        return transforms


class SemanticPrepareMixin(PrepareMixin):
    def get_pre_transforms(self, plan: dict) -> List[AbstractTransform]:
        """
        Search for unqiue instances -> Instances to Boxes

        Returns:
            List[AbstractTransform]: return a list of transformations

        Notes:
            make sure to call the super classes here!
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
        Map all instance to foreground

        Returns:
            List[AbstractTransform]: return a list of transformations

        Notes:
            make sure to call the super classes here!
        """
        trafos = super().get_pre_transforms(plan=plan)
        trafos.append(
            Instances2Fg(
                instance_key="target",
                seg_key="target_seg",
            )
        )
        return trafos
