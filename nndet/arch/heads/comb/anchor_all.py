from typing import Optional, Dict, List, Tuple

import torch
from torch import Tensor

from nndet.arch.heads.comb.base import AnchorHead
from nndet.arch.heads.classifier.dense import DenseClassifierType
from nndet.arch.heads.regressor.dense_single import DenseRegressorType

from nndet.core.boxes.coder import BoxCoderND


class BoxHeadAll(AnchorHead):
    def __init__(self,
                 classifier: DenseClassifierType,
                 regressor: DenseRegressorType,
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
        self.logger = None # get_logger(log_num_anchors) if log_num_anchors is not None else None

    def compute_loss(self,
                     prediction: Dict[str, Tensor],
                     target_labels: List[Tensor],
                     matched_gt_boxes: List[Tensor],
                     anchors: List[Tensor],
                     ) -> Tuple[Dict[str, Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute regression and classification loss
        N anchors over all images; M anchors per image => sum(M) = N

        Args:
            prediction: detection predictions for loss computation
                box_logits (Tensor): classification logits for each anchor
                    [N, num_classes]
                box_deltas (Tensor): offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            target_labels: target labels for each anchor (per image) [M]
            matched_gt_boxes: matched gt box for each anchor
                List[[N, dim *  2]], N=number of anchors per image
            anchors: anchors per image List[[N, dim *  2]]

        Returns:
            Tensor: dict with losses (reg for regression loss, cls for
                classification loss)
            Tensor: sampled positive indices of anchors (after concatenation)
            Tensor: sampled negative indices of anchors (after concatenation)
        """
        box_logits, box_deltas = prediction["box_logits"], prediction["box_deltas"]

        target_labels = torch.cat(target_labels, dim=0)
        batch_anchors = torch.cat(anchors, dim=0)
        pred_boxes = self.coder.decode_single(box_deltas, batch_anchors)
        target_boxes = torch.cat(matched_gt_boxes, dim=0)

        sampled_inds = torch.where(target_labels >= 0)[0]
        sampled_pos_inds = torch.where(target_labels >= 1)[0]

        losses = {}
        if sampled_pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                pred_boxes[sampled_pos_inds],
                target_boxes[sampled_pos_inds],
            ) / max(1, sampled_pos_inds.numel())

        losses["cls"] = self.classifier.compute_loss(
            box_logits[sampled_inds],
            target_labels[sampled_inds],
            ) / max(1, sampled_pos_inds.numel())
        # breakpoint()
        return losses, sampled_pos_inds, None
