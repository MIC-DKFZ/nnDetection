# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Tuple

import torch

from nndet.core.abstract import AbstractDetector, AbstractOneStageDetector
from nndet.core.rois.module import BaseRoIModule


class RCNN(AbstractDetector):
    def __init__(
        self,
        rpn: AbstractOneStageDetector,
        roi_module: BaseRoIModule,
    ) -> None:
        """
        Two stage detection module

        Args:
            rpn:
            roi_module:
        """
        super().__init__()
        self.rpn = rpn
        self.roi_module = roi_module

    def train_step(
        self,
        images: torch.Tensor,
        targets: dict,
        batch_num: int,
    ) -> Dict[str, torch.Tensor]:
        """
        #TODO docs2
        """
        # copy target classes
        targets["target_roi_classes"] = [trc.detach().clone() for trc in targets["target_classes"]]
        # map original targets to fg vs bg for RPN
        targets["target_classes"] = [torch.zeros_like(trc) for trc in targets["target_classes"]]

        losses, proposals, features = self.rpn.train_step_with_features(
            images=images,
            targets=targets,
            predict=True,
            batch_num=batch_num,
        )

        targets.pop("target_classes")  # remove targets to avoid accidental class mixup

        roi_losses = self.roi_module.train_step(
            images=images,
            features=features,
            proposals=proposals,
            targets=targets,
        )

        losses.update(roi_losses)
        return losses

    @torch.no_grad()
    def validation_step(
        self,
        images: torch.Tensor,
        targets: dict,
        batch_num: bool,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        predictions = self.inference_step(images=images)
        losses = {"placeholder": torch.tensor(0)}
        return losses, predictions

    @torch.no_grad()
    def inference_step(
        self,
        images: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        proposals, features = self.rpn.inference_step_with_features(images=images, **kwargs)
        predictions = {f"rpn_{key}": item for key, item in proposals.items()}

        roi_predictions = self.roi_module.inference_step(
            images=images,
            features=features,
            proposals=proposals,
        )
        predictions.update(roi_predictions)
        return predictions
