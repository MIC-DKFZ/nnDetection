from typing import Tuple, Dict, Any, Optional

import torch

from nndet.arch.abstract import AbstractModel
from nndet.core.rois.module import RoIModule
from nndet.utils.tensor import detach_all


class RCNN(AbstractModel):
    def __init__(self,
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
        #TODO
        """
        # print(targets["target_boxes"])
        losses, proposals, features = self.rpn.train_step_with_features(
            images=images,
            targets=targets,
            predict=True,
            batch_num=batch_num,
        )
        # do not propagate through proposals
        proposals = detach_all(proposals)

        roi_losses, roi_prediction = self.roi_module.train_step(
            images=images,
            features=features,
            proposals=proposals,
            targets=targets,
            predict=predict,
        )
        for key, item in roi_losses.items():
            losses[f"roi_{key}"] = item
        return losses, roi_prediction

    @torch.no_grad()
    def inference_step(
        self,
        images: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        proposals, features = self.rpn.inference_step_with_features(
            images=images,
            **kwargs
        )
        roi_prediction = self.roi_module.inference_step(
            images=images,
            features=features,
            proposals=proposals,
        )
        return roi_prediction


# class Sequencer(torch.nn.Module):
#     def __init__(self,
#                  roi_heads: List[RoIModuleType],
#                  ) -> None:
#         """
#         Cascade multiple RoI Heads

#         TODO: gradient scaling
#         TODO: detach proposals
#         """
#         super().__init__()
#         self.roi_heads = torch.nn.ModuleList(roi_heads)

#     def forward(self):
#         for head in self.roi_heads:
#             # predict rois
#             pass
#         pass
