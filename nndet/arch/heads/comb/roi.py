from typing import Optional, Dict, Tuple

import torch
from torch import Tensor

from nndet.arch.heads.comb.base import RoIHead

from nndet.core.boxes.coder import BoxCoderND


class RoIBoxHead(RoIHead):
    def __init__(
        self,
        classifier,  # : DenseClassifierType,
        regressor,  # : DenseRegressorType,
        coder: BoxCoderND,
        shared: Optional[torch.nn.Module] = None,
    ):
        """
        Box head with classifier and regression module. Uses all
        foreground anchors for regression an passes all anchors to classifier

        Args:
            classifier: classifier module
            regressor: regression module
            shared: optional shared module which is applied to before the
                classifier and regression head
        """
        super().__init__(
            classifier=classifier,
            regressor=regressor,
            coder=coder,
            shared=shared,
        )

    def compute_loss(
        self,
        prediction: Dict[str, Tensor],
        target_labels: Tensor,
        matched_gt_boxes: Tensor,
        proposals: Tensor,
    ) -> Tuple[Dict[str, Tensor], torch.Tensor, Optional[torch.Tensor]]:
        # TODO: might save additional computation by passing pos indices
        box_logits, box_deltas = prediction["box_logits"], prediction["box_deltas"]

        sampled_inds = torch.where(target_labels >= 0)[0]
        sampled_pos_inds = torch.where(target_labels >= 1)[0]

        target_deltas_sampled = self.coder.encode_single(
            matched_gt_boxes[sampled_pos_inds],
            proposals[sampled_pos_inds],
        )

        losses = {}
        if sampled_pos_inds.numel() > 0:
            losses["reg"] = (
                self.regressor.compute_loss(
                    box_deltas[sampled_pos_inds],
                    target_deltas_sampled,
                )
                / max(1, sampled_pos_inds.numel())
            )

        losses["cls"] = (
            self.classifier.compute_loss(
                box_logits[sampled_inds],
                target_labels[sampled_inds].long(),
            )
            / max(1, sampled_pos_inds.numel())
        )
        return losses, sampled_pos_inds, None
