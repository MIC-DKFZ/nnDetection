# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, Tuple

import torch
from loguru import logger
from torch import Tensor

from nndet.core.boxes.coder import BoxCoderND
from nndet.nn.heads.comb.base import RoIHead
from nndet.training.ema import EMABiasStepsModule
from nndet.utils.dist import get_world_size, is_dist_avail_and_initialized


class RoIBoxHead(RoIHead):
    def __init__(
        self,
        classifier,
        regressor,
        coder: BoxCoderND,
        shared: Optional[torch.nn.Module] = None,
        ema_loss_kwargs: Optional[Dict] = None,
    ):
        """
        Specific implementation of a box head with classifier and regression
        modules. Optionally, Exponential Moving Averages can be activated
        to smooth out the denominator of the loss functions.

        Args:
            classifier: classifier module
            regressor: regression module
            coder: Module to encoder/decoder box delta wrt to anchors/proposals
            shared: optional shared module which is applied to before the
                classifier and regression head
            ema_loss_kwargs: provide keyword arguments for EMA loss. If `None`,
                no EMA loss is used.
        """
        super().__init__(
            classifier=classifier,
            regressor=regressor,
            coder=coder,
            shared=shared,
        )
        if ema_loss_kwargs is not None:
            self.pos_ema = EMABiasStepsModule(**ema_loss_kwargs)
            self.all_ema = EMABiasStepsModule(**ema_loss_kwargs)
            logger.info(f"Using EMA norm loss in RoI Head: pos {self.pos_ema} all {self.all_ema}")
        else:
            self.pos_ema = None
            self.all_ema = None

        if is_dist_avail_and_initialized():
            logger.warning(
                "Distributed training with Hard Negative Mining is not implemented in a way which is "
                "fully compatible with distributed training and unexpecetd bahvior might arise."
            )

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

                ``'box_deltas'`` torch.Tensor
                    bounding box deltas of shape [N, (num_classes *) dim * 2],
                    where N=number of RoIs, dim=number of spatial dimensions,
                    and num_classes is the number of foreground classes.
                    num_classes is only used for class specific regression.

                ``'box_logits'`` torch.Tensor
                    classification logits [N, num_classes] where N is the
                    number of RoIs and num_classes is the number of foreground
                    classes

            matched_gt_labels: target labels for each proposal [N], where
                N is the number of RoIs (0 is background)
            matched_gt_boxes: matched gt box for each proposal
                [N, dim *  2], where N is the number of RoIs, and dim
                is the number of spatial dimensions
            proposal_boxes: concatenated and extended proposals with batch index
                (batch_idx, x1, y1, x2, y2, (z1, z2))[N, 1 + dim * 2],
                where N is the number of RoIs, and dim is the number
                of spatial dimensions

        Returns:
            Tensor: dict with losses (reg for regression loss, cls for
                classification loss)
            Tensor: sampled positive indices
            Tensor: None
        """
        box_logits, box_deltas = prediction["box_logits"], prediction["box_deltas"]

        sampled_inds = torch.where(matched_gt_labels >= 0)[0]
        sampled_pos_inds = torch.where(matched_gt_labels >= 1)[0]

        reg_pred_sampled, reg_target_sampled = self.get_reg_targets_by_mode(
            batch_anchors=proposal_boxes[sampled_pos_inds],
            batch_target_boxes=matched_gt_boxes[sampled_pos_inds],
            batch_pred_deltas=box_deltas[sampled_pos_inds],
        )
        target_labels_sampled = matched_gt_labels[sampled_pos_inds]

        _numel_all = sampled_inds.numel()
        _numel_pos = sampled_pos_inds.numel()

        # average number of boxes for norm in distributed setting
        _numel_all = torch.tensor([_numel_all], dtype=torch.float, device=box_logits.device)
        _numel_pos = torch.tensor([_numel_pos], dtype=torch.float, device=box_logits.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(_numel_all)
            torch.distributed.all_reduce(_numel_pos)
            _numel_all = _numel_all / get_world_size()
            _numel_pos = _numel_pos / get_world_size()

        if self.all_ema is not None:
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
