# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import List, Tuple

import torch


class MaskPostprocessing:
    def __init__(
        self,
        class_agnostic: bool = True,
        **kwargs,
    ) -> None:
        """
        Provides an abstract interface to postprocess a batch of masks
        from a detection model.

        Args:
            regress_class_agnostic: todo
        """
        super().__init__()
        self.class_agnostic = class_agnostic  # FIXME

    def process_batch(
        self,
        reps: List[torch.Tensor],
        probs: List[torch.Tensor],
        labels: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        all_reps, all_probs, all_labels = [], [], []
        for idx in range(len(reps)):
            if self.class_agnostic:
                _reps, _probs, _labels = self.process_image_class_agnostic(
                    img_reps=reps[idx],
                    img_probs=probs[idx],
                    img_labels=labels[idx],
                )
            else:
                _reps, _probs, _labels = self.process_image_per_class(
                    img_reps=reps[idx],
                    img_probs=probs[idx],
                    img_labels=labels[idx],
                )

            all_reps.append(_reps)
            all_probs.append(_probs)
            all_labels.append(_labels)
        return all_reps, all_probs, all_labels

    @abstractmethod
    def process_image_class_agnostic(
        self,
        img_reps: torch.Tensor,
        img_probs: torch.Tensor,
        img_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def process_image_per_class(
        self,
        img_reps: torch.Tensor,
        img_probs: torch.Tensor,
        img_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class NoMaskPostprocessing(MaskPostprocessing):
    def process_image_class_agnostic(
        self,
        img_reps: torch.Tensor,
        img_probs: torch.Tensor,
        img_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (img_reps, img_probs, img_labels)

    def process_image_per_class(
        self,
        img_reps: torch.Tensor,
        img_probs: torch.Tensor,
        img_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError
