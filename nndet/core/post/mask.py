# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import List, Tuple

import torch


class MaskPostprocessing:
    def __init__(
        self,
        num_classes: int,
        is_class_agnostic: bool = True,
        **kwargs,
    ) -> None:
        """
        Provides an abstract interface to postprocess a batch of masks
        from a detection model.

        Args:
            num_classes: number of foreground classes
            is_class_agnostic: indicate wheter there is one mask per object
                or one mask per object per class
            kwargs: placeholder for future compatbility
        """
        super().__init__()
        self.num_classes = num_classes
        self.is_class_agnostic = is_class_agnostic

    @abstractmethod
    def process_batch(
        self,
        reps: List[torch.Tensor],
        probs: List[torch.Tensor],
        labels: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Process a batch masks

        Args:
            reps: mask representations for predicted obejcts
                [N, (num_classes,), dims] where N is the Number of objects,
                dims are the spatials dimensions of the masks and num_classes
                is the number of foreground classes and is only present
                for class specific masks
            probs: associated predicted probabilties of masks [N], where
                N is the number of objects
            labels: associated predicted labels of masks [N], where N
                is the number of objects

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
                postprocessed masks of shape [N, dims], probabilities [N]
                and labels [N]
        """
        raise NotImplementedError


class NoMaskPostprocessing(MaskPostprocessing):
    def process_batch(
        self,
        reps: List[torch.Tensor],
        probs: List[torch.Tensor],
        labels: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        No postprocessing is applied

        Args:
            reps: mask representations for predicted obejcts
                [N, (num_classes,), dims] where N is the Number of objects,
                dims are the spatials dimensions of the masks and num_classes
                is the number of foreground classes and is only present
                for class specific masks
            probs: associated predicted probabilties of masks [N], where
                N is the number of objects
            labels: associated predicted labels of masks [N], where N
                is the number of objects

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
                postprocessed masks of shape [N, dims], probabilities [N]
                and labels [N]
        """
        return reps, probs, labels


# not used currently
# @abstractmethod
# def process_batch(
#     self,
#     reps: List[torch.Tensor],
#     probs: List[torch.Tensor],
#     labels: List[torch.Tensor],
# ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
#     all_reps, all_probs, all_labels = [], [], []
#     for idx in range(len(reps)):
#         if self.is_class_agnostic:
#             _reps, _probs, _labels = self.process_image_class_agnostic(
#                 img_reps=reps[idx],
#                 img_probs=probs[idx],
#                 img_labels=labels[idx],
#             )
#         else:
#             _reps, _probs, _labels = self.process_image_per_class(
#                 img_reps=reps[idx],
#                 img_probs=probs[idx],
#                 img_labels=labels[idx],
#             )

#         all_reps.append(_reps)
#         all_probs.append(_probs)
#         all_labels.append(_labels)
#     return all_reps, all_probs, all_labels

# @abstractmethod
# def process_image_class_agnostic(
#     self,
#     img_reps: torch.Tensor,
#     img_probs: torch.Tensor,
#     img_labels: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     raise NotImplementedError

# @abstractmethod
# def process_image_per_class(
#     self,
#     img_reps: torch.Tensor,
#     img_probs: torch.Tensor,
#     img_labels: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     raise NotImplementedError
