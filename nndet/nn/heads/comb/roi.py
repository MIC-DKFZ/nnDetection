# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, Tuple

import torch
from loguru import logger
from torch import Tensor

from nndet.core.boxes.coder import BoxCoderND
from nndet.nn.heads.comb.base import RoIHead
from nndet.training.ema import EMA


class RoIBoxHead(RoIHead):
    def __init__(
        self,
        classifier,
        regressor,
        coder: BoxCoderND,
        shared: Optional[torch.nn.Module] = None,
        ema_loss_norm: bool = False,
    ):
        """
        Box head with classifier and regression module. Uses all
        foreground anchors for regression an passes all anchors to classifier

        Args:
            classifier: classifier module
            regressor: regression module
            shared: optional shared module which is applied to before the
                classifier and regression head
            ema_loss_norm: use ema to normalize denominator of losses
        """
        super().__init__(
            classifier=classifier,
            regressor=regressor,
            coder=coder,
            shared=shared,
        )
        self.ema_loss_norm = ema_loss_norm
        if self.ema_loss_norm:
            logger.info("Using EMA norm loss in RoI Head")
            self.all_ema = EMA(beta=0.95, bias_correction=True)
            self.pos_ema = EMA(beta=0.95, bias_correction=True)

    def compute_loss(
        self,
        prediction: Dict[str, Tensor],
        matched_gt_labels: Tensor,
        matched_gt_boxes: Tensor,
        proposal_boxes: Tensor,
    ) -> Tuple[Dict[str, Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute regression and classification loss

        Args:
            prediction: detection predictions for loss computation

                ``"box_logits"`` (Tensor)
                    classification logits for each anchor [N, num_classes]

                ``"box_deltas"`` (Tensor)
                    offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N (, num_classes), dim * 2]

            matched_gt_labels: assigned classification label for each proposal
            matched_gt_boxes: matched gt box for each proposal
                List[[N, dim *  2]], N=number of anchors per image
            proposal_boxes: proposal from RPN [N, dim *  2]

        Returns:
            Tensor: dict with losses (reg for regression loss, cls
                for classification loss)
            Tensor: sampled positive indices of anchors (after concatenation)
            Optional[Tensor]: None
        """
        box_logits, box_deltas = prediction["box_logits"], prediction["box_deltas"]

        sampled_inds = torch.where(matched_gt_labels >= 0)[0]
        sampled_pos_inds = torch.where(matched_gt_labels >= 1)[0]

        reg_pred_sampled, reg_target_sampled = self.get_reg_by_mode(
            batch_anchors=proposal_boxes[sampled_pos_inds],
            batch_target_boxes=matched_gt_boxes[sampled_pos_inds],
            batch_pred_deltas=box_deltas[sampled_pos_inds],
        )
        target_labels_sampled = matched_gt_labels[sampled_pos_inds]

        _numel_all = sampled_inds.numel()
        _numel_pos = sampled_pos_inds.numel()
        if self.ema_loss_norm:
            self.all_ema.add(_numel_all)
            self.pos_ema.add(_numel_pos)
            _numel_all = self.all_ema.get()
            _numel_pos = self.pos_ema.get()

        losses = {}
        if sampled_pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                reg_pred_sampled,
                reg_target_sampled,
                target_labels_sampled,
            ) / max(1, _numel_pos)

        losses["cls"] = self.classifier.compute_loss(
            box_logits[sampled_inds],
            matched_gt_labels[sampled_inds],
        ) / max(1, _numel_all)
        return losses, sampled_pos_inds, None
