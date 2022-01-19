from typing import Any, Dict, Optional, Tuple

import torch

from nndet.arch.abstract import AbstractModel
from nndet.core.rois.module import RoIModule


class RCNN(AbstractModel):
    def __init__(
        self,
        rpn: AbstractModel,
        roi_module: RoIModule,
    ) -> None:
        super().__init__()
        self.rpn = rpn
        self.roi_module = roi_module

    def train_step(
        self,
        images: torch.Tensor,
        targets: dict,
        predict: bool,
        batch_num: int,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict]]:
        """
        #TODO docs2
        """
        # copy target classes
        targets["target_roi_classes"] = [
            trc.detach().clone() for trc in targets["target_classes"]
        ]
        # map original targets to fg vs bg for RPN
        targets["target_classes"] = [
            torch.zeros_like(trc) for trc in targets["target_classes"]
        ]

        losses, proposals, features = self.rpn.train_step_with_features(
            images=images,
            targets=targets,
            predict=True,
            batch_num=batch_num,
        )

        predictions = {f"rpn_{key}": item for key, item in proposals.items()}
        targets.pop("target_classes")  # remove targets to avoid accidental class mixup

        roi_losses, roi_predictions = self.roi_module.train_step(
            images=images,
            features=features,
            proposals=proposals,
            targets=targets,
            predict=predict,
        )

        losses.update(roi_losses)
        if roi_predictions is not None:
            predictions.update(roi_predictions)
        return losses, predictions

    @torch.no_grad()
    def inference_step(
        self,
        images: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        proposals, features = self.rpn.inference_step_with_features(
            images=images, **kwargs
        )
        predictions = {f"rpn_{key}": item for key, item in proposals.items()}

        roi_predictions = self.roi_module.inference_step(
            images=images,
            features=features,
            proposals=proposals,
        )
        predictions.update(roi_predictions)
        return predictions
