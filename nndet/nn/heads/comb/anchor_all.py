# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger
from torch import Tensor

from nndet.core.boxes.coder import BoxCoderND
from nndet.nn.heads.classifier.dense import DenseClassifier
from nndet.nn.heads.comb.base import AnchorHead
from nndet.nn.heads.regressor.dense import DenseRegressor
from nndet.training.ema import EMA


class BoxHeadAll(AnchorHead):
    def __init__(
        self,
        classifier: DenseClassifier,
        regressor: DenseRegressor,
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
            logger.info("Using EMA norm loss in RPN Head")
            self.pos_ema = EMA(beta=0.95, bias_correction=True)
        self.logger = None  # get_logger(log_num_anchors) if log_num_anchors is not None else None

    def compute_loss(
        self,
        prediction: Dict[str, Tensor],
        matched_gt_labels: List[Tensor],
        matched_gt_boxes: List[Tensor],
        anchors: List[Tensor],
    ) -> Tuple[Dict[str, Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute regression and classification loss
        N anchors over all images; M anchors per image => sum(M) = N

        Args:
            prediction: detection predictions for loss computation

                ``'box_deltas'`` torch.Tensor
                    bounding box deltas of shape [N, (num_classes *) dim * 2],
                    where N=number of anchors, dim=number of spatial dimensions,
                    and num_classes is the number of foreground classes.
                    num_classes is only used for class specific regression.

                ``'box_logits'`` torch.Tensor
                    classification logits [N, num_classes] where N is the
                    number of anchors and num_classes is the number of
                    foreground classes

            matched_gt_labels: target labels for each anchor (per image) [M]
                where M is the number of anchors per image
            matched_gt_boxes: matched gt box for each anchor
                List[[M, dim *  2]], where M is the number of anchors per
                image and dim is the number of spatial dimensions
            anchors: anchors per image List[[M, dim *  2]], where M is the
                number of anchors per image and dim is the number of
                spatial dimensions

        Returns:
            Tensor: dict with losses (reg for regression loss, cls for
                classification loss)
            Tensor: sampled positive indices of anchors
                (after concatenation if sampled otherwise None)
            Tensor: sampled negative indices of anchors
                (after concatenation, if sampled otherwise None)
        """
        box_logits, box_deltas = prediction["box_logits"], prediction["box_deltas"]

        batch_anchors = torch.cat(anchors, dim=0)
        target_labels = torch.cat(matched_gt_labels, dim=0)
        target_boxes = torch.cat(matched_gt_boxes, dim=0)

        reg_pred, reg_target = self.get_reg_targets_by_mode(
            batch_anchors=batch_anchors,
            batch_target_boxes=target_boxes,
            batch_pred_deltas=box_deltas,
        )
        sampled_inds = torch.where(target_labels >= 0)[0]
        sampled_pos_inds = torch.where(target_labels >= 1)[0]

        _numel_pos = sampled_pos_inds.numel()
        if self.ema_loss_norm:
            self.pos_ema.add(_numel_pos)
            _numel_pos = self.pos_ema.get()

        losses = {}
        if sampled_pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                reg_pred[sampled_pos_inds],
                reg_target[sampled_pos_inds],
                target_labels[sampled_pos_inds],
            ) / max(1, _numel_pos)

        losses["cls"] = self.classifier.compute_loss(
            box_logits[sampled_inds],
            target_labels[sampled_inds],
        ) / max(1, _numel_pos)
        return losses, sampled_pos_inds, None
