# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger
from torch import Tensor

from nndet.core.boxes.coder import BoxCoderND
from nndet.core.boxes.sampler import AbstractSampler
from nndet.nn.heads.classifier.dense import DenseClassifier
from nndet.nn.heads.comb.base import AnchorHead
from nndet.nn.heads.regressor.dense import DenseRegressor
from nndet.training.ema import EMABiasStepsModule
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
        Notes:
            Regression loss will be normalized automatically while
            classification loss is expected to be normalized.
        """
        super().__init__(
            classifier=classifier,
            regressor=regressor,
            coder=coder,
            shared=shared,
        )

        self.logger = None  # get_logger(log_num_anchors) if log_num_anchors is not None else None
        self.fg_bg_sampler = sampler

    def compute_loss(
        self,
        prediction: Dict[str, Tensor],
        matched_gt_labels: List[Tensor],
        matched_gt_boxes: List[Tensor],
        anchors: List[Tensor],
    ) -> Tuple[Dict[str, Tensor], torch.Tensor, torch.Tensor]:
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
                where M is the number of anchors per image  (0 is background)
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

        losses = {}
        sampled_pos_inds, sampled_neg_inds = self.select_indices(matched_gt_labels, box_logits)
        sampled_inds = cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        batch_anchors = cat(anchors, dim=0)
        target_labels = cat(matched_gt_labels, dim=0)
        target_boxes = cat(matched_gt_boxes, dim=0)

        reg_pred_sampled, reg_target_sampled = self.get_reg_targets_by_mode(
            batch_anchors=batch_anchors[sampled_pos_inds],
            batch_target_boxes=target_boxes[sampled_pos_inds],
            batch_pred_deltas=box_deltas[sampled_pos_inds],
        )
        target_labels_sampled = target_labels[sampled_pos_inds]

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
                target_labels_sampled,
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
        ema_loss_kwargs: Optional[Dict] = None,
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
            ema_loss_kwargs: provide keyword arguments for EMA loss. If `None`,
                no EMA loss is used.

        Notes:
            Classification and Regression loss will be normalized by head
            automatically.
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
            logger.info(f"Using EMA norm loss in RPN Head: pos {self.pos_ema} all {self.all_ema}")
        else:
            self.pos_ema = None
            self.all_ema = None

        self.logger = None  # get_logger(log_num_anchors) if log_num_anchors is not None else None
        self.fg_bg_sampler = sampler

    def compute_loss(
        self,
        prediction: Dict[str, Tensor],
        matched_gt_labels: List[Tensor],
        matched_gt_boxes: List[Tensor],
        anchors: List[Tensor],
    ) -> Tuple[Dict[str, Tensor], torch.Tensor, torch.Tensor]:
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
                where M is the number of anchors per image  (0 is background)
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

        losses = {}
        sampled_pos_inds, sampled_neg_inds = self.select_indices(matched_gt_labels, box_logits)
        sampled_inds = cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        batch_anchors = cat(anchors, dim=0)
        target_labels = cat(matched_gt_labels, dim=0)
        target_boxes = cat(matched_gt_boxes, dim=0)

        reg_pred_sampled, reg_target_sampled = self.get_reg_targets_by_mode(
            batch_anchors=batch_anchors[sampled_pos_inds],
            batch_target_boxes=target_boxes[sampled_pos_inds],
            batch_pred_deltas=box_deltas[sampled_pos_inds],
        )
        target_labels_sampled = target_labels[sampled_pos_inds]

        # target_deltas = self.coder.encode(matched_gt_boxes, anchors)
        # target_deltas_sampled = torch.cat(target_deltas, dim=0)[sampled_pos_inds]

        # assert len(batch_anchors) == len(batch_matched_gt_boxes)
        # assert len(batch_anchors) == len(box_deltas)
        # assert len(batch_anchors) == len(box_logits)
        # assert len(batch_anchors) == len(target_labels)

        _numel_all = sampled_inds.numel()
        _numel_pos = sampled_pos_inds.numel()
        if self.all_ema is not None:
            self.all_ema.add(_numel_all)
            self.pos_ema.add(_numel_pos)
            _numel_all = self.all_ema.get()
            _numel_pos = self.pos_ema.get()

        if sampled_pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                reg_pred_sampled,
                reg_target_sampled,
                target_labels_sampled,
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
        matched_gt_labels: List[Tensor],
        matched_gt_boxes: List[Tensor],
        anchors: List[Tensor],
    ) -> Tuple[Dict[str, Tensor], torch.Tensor, torch.Tensor]:
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
                where M is the number of anchors per image  (0 is background)
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

        losses = {}
        sampled_pos_inds, sampled_neg_inds = self.select_indices(matched_gt_labels, box_logits)
        sampled_inds = cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        target_labels = cat(matched_gt_labels, dim=0)

        losses["cls"] = self.classifier.compute_loss(box_logits[sampled_inds], target_labels[sampled_inds])

        pos_inds = torch.where(target_labels >= 1)[0]
        batch_anchors = cat(anchors, dim=0)
        target_boxes = cat(matched_gt_boxes, dim=0)

        reg_pred_sampled, reg_target_sampled = self.get_reg_targets_by_mode(
            batch_anchors=batch_anchors[pos_inds],
            batch_target_boxes=target_boxes[pos_inds],
            batch_pred_deltas=box_deltas[pos_inds],
        )
        target_labels_sampled = target_labels[sampled_pos_inds]

        # assert len(batch_anchors) == len(batch_matched_gt_boxes)
        # assert len(batch_anchors) == len(box_deltas)
        # assert len(batch_anchors) == len(box_logits)
        # assert len(batch_anchors) == len(target_labels)

        if pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                reg_pred_sampled,
                reg_target_sampled,
                target_labels_sampled,
            ) / max(1, pos_inds.numel())

        return losses, sampled_pos_inds, sampled_neg_inds


class BoxHeadHNMDualReg(BoxHeadHNM):
    def __init__(
        self,
        classifier: DenseClassifier,
        regressor: DenseRegressor,
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
        if not self.regressor.get_reg_mode() == BoxRegressionMode.DUAL:
            raise ValueError(
                f"Provided regressor {self.regressor.__class__.__name__} "
                f"with reg mode {self.regressor.get_reg_mode()} is not compatible "
                f"with {self.__class__.__name__} which requires 'dual' reg mode"
            )

    def compute_loss(
        self,
        prediction: Dict[str, Tensor],
        matched_gt_labels: List[Tensor],
        matched_gt_boxes: List[Tensor],
        anchors: List[Tensor],
    ) -> Tuple[Dict[str, Tensor], torch.Tensor, torch.Tensor]:
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
                where M is the number of anchors per image (0 is background)
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

        losses = {}
        sampled_pos_inds, sampled_neg_inds = self.select_indices(matched_gt_labels, box_logits)
        sampled_inds = cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        target_labels = cat(matched_gt_labels, dim=0)

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
                target_label=target_labels[sampled_pos_inds],
            ) / max(1, sampled_pos_inds.numel())

        return losses, sampled_pos_inds, sampled_neg_inds


################################################################################
# Deprecations
from nndet.utils.info import deprecate


class BoxHeadHNMNative(BoxHeadHNM):
    @deprecate(
        replacement="`BoxHeadHNM`",
        deprecate="v0.1.2",
    )
    def __init__(self, *args, **kwargs):
        """ """
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        prediction: Dict[str, Tensor],
        matched_gt_labels: List[Tensor],
        matched_gt_boxes: List[Tensor],
        anchors: List[Tensor],
    ) -> Tuple[Dict[str, Tensor], torch.Tensor, torch.Tensor]:
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
                where M is the number of anchors per image (0 is background)
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

        losses = {}
        # with torch.no_grad():
        sampled_pos_inds, sampled_neg_inds = self.select_indices(matched_gt_labels, box_logits)
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        target_labels = torch.cat(matched_gt_labels, dim=0)
        batch_anchors = torch.cat(anchors, dim=0)
        pred_boxes_sampled = self.coder.decode_single(box_deltas[sampled_pos_inds], batch_anchors[sampled_pos_inds])

        target_boxes_sampled = torch.cat(matched_gt_boxes, dim=0)[sampled_pos_inds]
        target_labels_sampled = target_labels[sampled_pos_inds]
        if sampled_pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                pred_boxes_sampled,
                target_boxes_sampled,
                target_labels_sampled,
            ) / max(1, sampled_pos_inds.numel())

        losses["cls"] = self.classifier.compute_loss(box_logits[sampled_inds], target_labels[sampled_inds])
        return losses, sampled_pos_inds, sampled_neg_inds
