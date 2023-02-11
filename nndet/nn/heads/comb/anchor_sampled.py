# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Tuple, Union

import torch
from loguru import logger
from torch import Tensor

from nndet.core.boxes.coder import BoxCoderND
from nndet.core.boxes.sampler import AbstractSampler
from nndet.nn.heads.abstract import ClassifierType, RegressorType
from nndet.nn.heads.classifier.dense import DenseClassifier
from nndet.nn.heads.comb.base import AnchorHead
from nndet.nn.heads.regressor.dense import DenseRegressor
from nndet.training.ema import EMA
from nndet.utils.enums import BoxRegressionMode
from nndet.utils.tensor import cat


class BoxHeadHNM(AnchorHead):
    def __init__(
        self,
        classifier: DenseClassifier,
        regressor: DenseRegressor,
        coder: BoxCoderND,
        sampler: AbstractSampler,
        shared: Optional[torch.nn.Module] = None,
        reg_mode: Union[str, BoxRegressionMode] = "encode",
    ):
        """
        Box detection head with classifier and regression module.
        Uses hard negative example mining to compute loss

        Args:
            classifier: classifier module
            regressor: regression module
            coder: Module to encoder/decoder box delta wrt to anchors/proposals
            sampler: sampler for select positive and negative examples
            shared: optional shared module which is applied to before the
                classifier and regression head
            reg_mode: define regression mode. One of `decode` | `encode`

                ``"decode"``: uses the predicted box deltas to decode the
                    predicted boxes which are passed to the regression loss
                    in combination with the matched ground truth boxes

                ``"encode"``: uses the matched ground truth to encode the
                    expected box deltas which are passed to the regression loss
                    in combination with the predicted box deltas

        Notes:
            Regression loss will be normalized automatically while
            classification loss is expected to be normalized.
        """
        super().__init__(
            classifier=classifier,
            regressor=regressor,
            coder=coder,
            shared=shared,
            reg_mode=reg_mode,
        )

        self.logger = None  # get_logger(log_num_anchors) if log_num_anchors is not None else None
        self.fg_bg_sampler = sampler

    def compute_loss(
        self,
        prediction: Dict[str, Tensor],
        target_labels: List[Tensor],
        matched_gt_boxes: List[Tensor],
        anchors: List[Tensor],
    ) -> Tuple[Dict[str, Tensor], torch.Tensor, torch.Tensor]:
        """
        Compute regression and classification loss
        N anchors over all images; M anchors per image => sum(M) = N

        Args:
            prediction: detection predictions for loss computation

                ``"box_logits"`` (Tensor)
                    classification logits for each anchor [N, num_classes]

                ``"box_deltas"`` (Tensor)
                    offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]

            target_labels (List[Tensor]): target labels for each anchor
                (per image) [M]
            matched_gt_boxes: matched gt box for each anchor
                List[[N, dim *  2]], N=number of anchors per image
            anchors: anchors per image List[[N, dim *  2]]

        Returns:
            Tensor: dict with losses (reg for regression loss, cls
                for classification loss)
            Tensor: sampled positive indices of anchors (after concatenation)
            Tensor: sampled negative indices of anchors (after concatenation)
        """
        box_logits, box_deltas = prediction["box_logits"], prediction["box_deltas"]

        losses = {}
        sampled_pos_inds, sampled_neg_inds = self.select_indices(target_labels, box_logits)
        sampled_inds = cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        batch_anchors = cat(anchors, dim=0)
        target_labels = cat(target_labels, dim=0)
        target_boxes = cat(matched_gt_boxes, dim=0)

        reg_pred_sampled, reg_target_sampled = self.get_reg_by_mode(
            batch_anchors=batch_anchors[sampled_pos_inds],
            batch_target_boxes=target_boxes[sampled_pos_inds],
            batch_pred_deltas=box_deltas[sampled_pos_inds],
        )

        # target_deltas = self.coder.encode(matched_gt_boxes, anchors)
        # target_deltas_sampled = torch.cat(target_deltas, dim=0)[sampled_pos_inds]

        # assert len(batch_anchors) == len(batch_matched_gt_boxes)
        # assert len(batch_anchors) == len(box_deltas)
        # assert len(batch_anchors) == len(box_logits)
        # assert len(batch_anchors) == len(target_labels)

        if sampled_pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                reg_pred_sampled,
                reg_target_sampled,
            ) / max(1, sampled_pos_inds.numel())

        losses["cls"] = self.classifier.compute_loss(box_logits[sampled_inds], target_labels[sampled_inds])
        return losses, sampled_pos_inds, sampled_neg_inds

    def select_indices(
        self,
        target_labels: List[Tensor],
        boxes_scores: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample positive and negative anchors from target labels

        Args:
            target_labels (List[Tensor]): target labels for each anchor
                (per image) [M]
            boxes_scores (Tensor): classification logits for each anchor
                [N, num_classes]

        Returns:
            Tensor: sampled positive indices [R]
            Tensor: sampled negative indices [R]
        """
        boxes_max_fg_probs = self.classifier.logits_to_probs(boxes_scores)
        boxes_max_fg_probs = boxes_max_fg_probs.max(dim=1)[0]  # search max of fg probs

        # positive and negative anchor indices per image
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(target_labels, boxes_max_fg_probs)
        sampled_pos_inds = torch.where(cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(cat(sampled_neg_inds, dim=0))[0]

        # if self.logger:
        #     self.logger.add_scalar("train/num_pos", sampled_pos_inds.numel())
        #     self.logger.add_scalar("train/num_neg", sampled_neg_inds.numel())

        return sampled_pos_inds, sampled_neg_inds


class BoxHeadHNMV2(AnchorHead):
    def __init__(
        self,
        classifier: DenseClassifier,
        regressor: DenseRegressor,
        coder: BoxCoderND,
        sampler: AbstractSampler,
        shared: Optional[torch.nn.Module] = None,
        reg_mode: Union[str, BoxRegressionMode] = "encode",
        ema_loss_norm: bool = False,
    ):
        """
        Box detection head with classifier and regression module.
        Uses hard negative example mining to compute loss. Optionally,
        the normalization of the loss functions can be with EMA.

        Args:
            classifier: classifier module
            regressor: regression module
            coder: Module to encoder/decoder box delta wrt to anchors/proposals
            sampler: sampler for select positive and negative examples
            shared: optional shared module which is applied to before the
                classifier and regression head
            reg_mode: define regression mode. One of `decode` | `encode`

                ``"decode"``: uses the predicted box deltas to decode the
                    predicted boxes which are passed to the regression loss
                    in combination with the matched ground truth boxes

                `"encode"`: uses the matched ground truth to encode the
                    expected box deltas which are passed to the regression loss
                    in combination with the predicted box deltas

            ema_loss_norm: use ema to normalize denominator of losses

        Notes:
            Classification and Regression loss will be normalized by head
            automatically.
        """
        super().__init__(
            classifier=classifier,
            regressor=regressor,
            coder=coder,
            shared=shared,
            reg_mode=reg_mode,
        )
        self.ema_loss_norm = ema_loss_norm
        if self.ema_loss_norm:
            logger.info("Using EMA norm loss in Dense Anchor Head")
            self.all_ema = EMA(beta=0.95, bias_correction=True)
            self.pos_ema = EMA(beta=0.95, bias_correction=True)

        self.logger = None  # get_logger(log_num_anchors) if log_num_anchors is not None else None
        self.fg_bg_sampler = sampler

    def compute_loss(
        self,
        prediction: Dict[str, Tensor],
        target_labels: List[Tensor],
        matched_gt_boxes: List[Tensor],
        anchors: List[Tensor],
    ) -> Tuple[Dict[str, Tensor], torch.Tensor, torch.Tensor]:
        """
        Compute regression and classification loss
        N anchors over all images; M anchors per image => sum(M) = N

        Args:
            prediction: detection predictions for loss computation

                ``"box_logits"`` (Tensor): classification logits for each anchor
                    [N, num_classes]

                ``"box_deltas"`` (Tensor): offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]

            target_labels (List[Tensor]): target labels for each anchor
                (per image) [M]
            matched_gt_boxes: matched gt box for each anchor
                List[[N, dim *  2]], N=number of anchors per image
            anchors: anchors per image List[[N, dim *  2]]

        Returns:
            Tensor: dict with losses (reg for regression loss, cls
                for classification loss)
            Tensor: sampled positive indices of anchors (after concatenation)
            Tensor: sampled negative indices of anchors (after concatenation)
        """
        box_logits, box_deltas = prediction["box_logits"], prediction["box_deltas"]

        losses = {}
        sampled_pos_inds, sampled_neg_inds = self.select_indices(target_labels, box_logits)
        sampled_inds = cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        batch_anchors = cat(anchors, dim=0)
        target_labels = cat(target_labels, dim=0)
        target_boxes = cat(matched_gt_boxes, dim=0)

        reg_pred_sampled, reg_target_sampled = self.get_reg_by_mode(
            batch_anchors=batch_anchors[sampled_pos_inds],
            batch_target_boxes=target_boxes[sampled_pos_inds],
            batch_pred_deltas=box_deltas[sampled_pos_inds],
        )

        # target_deltas = self.coder.encode(matched_gt_boxes, anchors)
        # target_deltas_sampled = torch.cat(target_deltas, dim=0)[sampled_pos_inds]

        # assert len(batch_anchors) == len(batch_matched_gt_boxes)
        # assert len(batch_anchors) == len(box_deltas)
        # assert len(batch_anchors) == len(box_logits)
        # assert len(batch_anchors) == len(target_labels)

        _numel_all = sampled_inds.numel()
        _numel_pos = sampled_pos_inds.numel()
        if self.ema_loss_norm:
            self.all_ema.add(_numel_all)
            self.pos_ema.add(_numel_pos)
            _numel_all = self.all_ema.get()
            _numel_pos = self.pos_ema.get()

        if sampled_pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                reg_pred_sampled,
                reg_target_sampled,
            ) / max(1, _numel_pos)

        losses["cls"] = self.classifier.compute_loss(
            box_logits[sampled_inds],
            target_labels[sampled_inds],
        ) / max(1, _numel_all)
        return losses, sampled_pos_inds, sampled_neg_inds

    def select_indices(
        self,
        target_labels: List[Tensor],
        boxes_scores: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample positive and negative anchors from target labels

        Args:
            target_labels (List[Tensor]): target labels for each anchor
                (per image) [M]
            boxes_scores (Tensor): classification logits for each anchor
                [N, num_classes]

        Returns:
            Tensor: sampled positive indices [R]
            Tensor: sampled negative indices [R]
        """
        boxes_max_fg_probs = self.classifier.logits_to_probs(boxes_scores)
        boxes_max_fg_probs = boxes_max_fg_probs.max(dim=1)[0]  # search max of fg probs

        # positive and negative anchor indices per image
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(target_labels, boxes_max_fg_probs)
        sampled_pos_inds = torch.where(cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(cat(sampled_neg_inds, dim=0))[0]

        # if self.logger:
        #     self.logger.add_scalar("train/num_pos", sampled_pos_inds.numel())
        #     self.logger.add_scalar("train/num_neg", sampled_neg_inds.numel())

        return sampled_pos_inds, sampled_neg_inds


class BoxHeadHNMRegAll(BoxHeadHNM):
    def compute_loss(
        self,
        prediction: Dict[str, Tensor],
        target_labels: List[Tensor],
        matched_gt_boxes: List[Tensor],
        anchors: List[Tensor],
    ) -> Tuple[Dict[str, Tensor], torch.Tensor, torch.Tensor]:
        """
        Compute regression and classification loss
        N anchors over all images; M anchors per image => sum(M) = N

        Args:
            prediction: detection predictions for loss computation

                ``'box_logits'`` (Tensor)
                    classification logits for each anchor [N, num_classes]

                ``"box_deltas"`` (Tensor)
                    offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]

            target_labels (List[Tensor]): target labels for each anchor
                (per image) [M]
            matched_gt_boxes: matched gt box for each anchor
                List[[N, dim *  2]], N=number of anchors per image
            anchors: anchors per image List[[N, dim *  2]]

        Returns:
            Tensor: dict with losses (reg for regression loss, cls
                for classification loss)
            Tensor: sampled positive indices of anchors (after concatenation)
            Tensor: sampled negative indices of anchors (after concatenation)
        """
        box_logits, box_deltas = prediction["box_logits"], prediction["box_deltas"]

        losses = {}
        sampled_pos_inds, sampled_neg_inds = self.select_indices(target_labels, box_logits)
        sampled_inds = cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        target_labels = cat(target_labels, dim=0)

        losses["cls"] = self.classifier.compute_loss(box_logits[sampled_inds], target_labels[sampled_inds])

        pos_inds = torch.where(target_labels >= 1)[0]
        batch_anchors = cat(anchors, dim=0)
        target_boxes = cat(matched_gt_boxes, dim=0)

        reg_pred_sampled, reg_target_sampled = self.get_reg_by_mode(
            batch_anchors=batch_anchors[pos_inds],
            batch_target_boxes=target_boxes[pos_inds],
            batch_pred_deltas=box_deltas[pos_inds],
        )

        # assert len(batch_anchors) == len(batch_matched_gt_boxes)
        # assert len(batch_anchors) == len(box_deltas)
        # assert len(batch_anchors) == len(box_logits)
        # assert len(batch_anchors) == len(target_labels)

        if pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                reg_pred_sampled,
                reg_target_sampled,
            ) / max(1, pos_inds.numel())

        return losses, sampled_pos_inds, sampled_neg_inds


class BoxHeadHNMDualReg(BoxHeadHNM):
    def __init__(
        self,
        classifier: ClassifierType,
        regressor: RegressorType,
        coder: BoxCoderND,
        shared: Optional[torch.nn.Module] = None,
    ):
        """
        Can be used to compute regression lossed on encoded and decoded
        predictions simultaniously (e.g. L1 + GIoU Loss)

        Args:
            classifier: classifier module
            regressor: regression module
            coder: Module to encoder/decoder box delta wrt to anchors/proposals
            shared: optional shared module which is applied to before the
                classifier and regression head
        """
        super().__init__()
        self.classifier = classifier
        self.regressor = regressor
        self.shared = shared
        self.coder = coder

    def compute_loss(
        self,
        prediction: Dict[str, Tensor],
        target_labels: List[Tensor],
        matched_gt_boxes: List[Tensor],
        anchors: List[Tensor],
    ) -> Tuple[Dict[str, Tensor], torch.Tensor, torch.Tensor]:
        """
        Compute regression and classification loss
        N anchors over all images; M anchors per image => sum(M) = N

        Args:
            prediction: detection predictions for loss computation

                ``'box_logits'`` (Tensor)
                    classification logits for each anchor [N, num_classes]

                ``'box_deltas'`` (Tensor)
                    offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]

            target_labels (List[Tensor]): target labels for each anchor
                (per image) [M]
            matched_gt_boxes: matched gt box for each anchor
                List[[N, dim *  2]], N=number of anchors per image
            anchors: anchors per image List[[N, dim *  2]]

        Returns:
            Tensor: dict with losses (reg for regression loss, cls
                for classification loss)
            Tensor: sampled positive indices of anchors (after concatenation)
            Tensor: sampled negative indices of anchors (after concatenation)
        """
        box_logits, box_deltas = prediction["box_logits"], prediction["box_deltas"]

        losses = {}
        sampled_pos_inds, sampled_neg_inds = self.select_indices(target_labels, box_logits)
        sampled_inds = cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        target_labels = cat(target_labels, dim=0)

        batch_matched_gt_boxes = cat(matched_gt_boxes, dim=0)
        batch_anchors = cat(anchors, dim=0)

        # encode anchor deltas
        target_deltas_sampled = self.coder.encode_single(
            batch_matched_gt_boxes[sampled_pos_inds],
            batch_anchors[sampled_pos_inds],
        )
        # decode prediction boxes
        pred_boxes_sampled = self.coder.decode_single(box_deltas[sampled_pos_inds], batch_anchors[sampled_pos_inds])

        # compute losses
        losses["cls"] = self.classifier.compute_loss(box_logits[sampled_inds], target_labels[sampled_inds])

        if sampled_pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                pred_deltas=box_deltas[sampled_pos_inds],
                target_deltas=target_deltas_sampled,
                pred_boxes=pred_boxes_sampled,
                target_boxes=batch_matched_gt_boxes[sampled_pos_inds],
            ) / max(1, sampled_pos_inds.numel())

        return losses, sampled_pos_inds, sampled_neg_inds


################################################################################
# Deprecations
from nndet.utils.info import deprecate


class BoxHeadHNMNative(BoxHeadHNM):
    @deprecate(
        replacement="`BoxHeadHNM` with `reg_mode=decode`",
        deprecate="v0.1.2",
        remove="v0.2",
    )
    def __init__(self, *args, **kwargs):
        """ """
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        prediction: Dict[str, Tensor],
        target_labels: List[Tensor],
        matched_gt_boxes: List[Tensor],
        anchors: List[Tensor],
    ) -> Tuple[Dict[str, Tensor], torch.Tensor, torch.Tensor]:
        """
        Compute regression and classification loss
        N anchors over all images; M anchors per image => sum(M) = N

        This head decodes the relative offsets from the networks and computes
        the regression loss directly on the bounding boxes (e.g. for GIoU loss)

        Args:
            prediction: detection predictions for loss computation
            target_labels (List[Tensor]): target labels for each anchor
                (per image) [M]
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

        losses = {}
        # with torch.no_grad():
        sampled_pos_inds, sampled_neg_inds = self.select_indices(target_labels, box_logits)
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        target_labels = torch.cat(target_labels, dim=0)
        batch_anchors = torch.cat(anchors, dim=0)
        pred_boxes_sampled = self.coder.decode_single(box_deltas[sampled_pos_inds], batch_anchors[sampled_pos_inds])

        target_boxes_sampled = torch.cat(matched_gt_boxes, dim=0)[sampled_pos_inds]
        if sampled_pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                pred_boxes_sampled,
                target_boxes_sampled,
            ) / max(1, sampled_pos_inds.numel())

        losses["cls"] = self.classifier.compute_loss(box_logits[sampled_inds], target_labels[sampled_inds])
        return losses, sampled_pos_inds, sampled_neg_inds


class BoxHeadHNMNativeRegAll(BoxHeadHNM):
    @deprecate(
        replacement="`BoxHeadHNMRegAll` with `reg_mode=decode`",
        deprecate="v0.1.2",
        remove="v0.2",
    )
    def __init__(self, *args, **kwargs):
        """ """
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        prediction: Dict[str, Tensor],
        target_labels: List[Tensor],
        matched_gt_boxes: List[Tensor],
        anchors: List[Tensor],
    ) -> Tuple[Dict[str, Tensor], torch.Tensor, torch.Tensor]:
        """
        Compute regression and classification loss
        N anchors over all images; M anchors per image => sum(M) = N

        This head decodes the relative offsets from the networks and computes
        the regression loss directly on the bounding boxes (e.g. for GIoU loss)

        Args:
            prediction: detection predictions for loss computation
            target_labels (List[Tensor]): target labels for each anchor
                (per image) [M]
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

        losses = {}
        sampled_pos_inds, sampled_neg_inds = self.select_indices(target_labels, box_logits)
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        target_labels = torch.cat(target_labels, dim=0)
        batch_anchors = torch.cat(anchors, dim=0)

        assert len(batch_anchors) == len(box_deltas)
        assert len(batch_anchors) == len(box_logits)
        assert len(batch_anchors) == len(target_labels)

        losses["cls"] = self.classifier.compute_loss(box_logits[sampled_inds], target_labels[sampled_inds])

        pos_inds = torch.where(target_labels >= 1)[0]
        pred_boxes = self.coder.decode_single(box_deltas[pos_inds], batch_anchors[pos_inds])
        target_boxes = torch.cat(matched_gt_boxes, dim=0)[pos_inds]

        if pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                pred_boxes,
                target_boxes,
            ) / max(1, pos_inds.numel())

        return losses, sampled_pos_inds, sampled_neg_inds
