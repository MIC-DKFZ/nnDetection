"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn as nn

from torch import Tensor
from typing import Dict, List, Tuple, Optional, TypeVar
from abc import abstractmethod

from nndet.detection.boxes import BoxCoderND
from nndet.detection.boxes.sampler import AbstractSampler
from nndet.models.heads.classifier import Classifier
from nndet.models.heads.regressor import Regressor


class AbstractHead(nn.Module):
    """
    Provides an abstract interface for an module which takes
    inputs and computed its own loss
    """
    @abstractmethod
    def forward(self, x: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute forward pass
        
        Args
            x: feature maps
        """
        raise NotImplementedError 

    @abstractmethod
    def compute_loss(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute loss
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess_for_inference(self,
                                  prediction: Dict[str, torch.Tensor],
                                  *args, **kwargs,
                                  ) -> Dict[str, torch.Tensor]:
        """
        Postprocess predictions for inference e.g. ocnvert logits to probs

        Args:
            Dict[str, torch.Tensor]: predictions from this head
            List[torch.Tensor]: anchors per image
        """
        raise NotImplementedError


class DetectionHead(AbstractHead):
    def __init__(self,
                 classifier: Classifier,
                 regressor: Regressor,
                 coder: BoxCoderND,
                 ):
        """
        Detection head with classifier and regression module
        
        Args:
            classifier: classifier module
            regressor: regression module
        """
        super().__init__()
        self.classifier = classifier
        self.regressor = regressor
        self.coder = coder

    def forward(self,
                fmaps: List[torch.Tensor],
                ) -> Dict[str, torch.Tensor]:
        """
        Forward feature maps through head modules

        Args:
            fmaps: list of feature maps for head module

        Returns:
            Dict[str, torch.Tensor]: predictions
                `box_deltas`(Tensor): bounding box offsets
                    [Num_Anchors_Batch, (dim * 2)]
                `box_logits`(Tensor): classification logits
                    [Num_Anchors_Batch, (num_classes)]
        """
        logits, offsets = [], []
        for level, p in enumerate(fmaps):
            logits.append(self.classifier(p, level=level))
            offsets.append(self.regressor(p, level=level))

        sdim = fmaps[0].ndim - 2
        box_deltas = torch.cat(offsets, dim=1).reshape(-1, sdim * 2)
        box_logits = torch.cat(logits, dim=1).flatten(0, -2)
        return {"box_deltas": box_deltas, "box_logits": box_logits}

    @abstractmethod
    def compute_loss(self,
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
                `box_logits`: classification logits for each anchor [N]
                `box_deltas`: offsets for each anchor
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
        raise NotImplementedError

    def postprocess_for_inference(self,
                                  prediction: Dict[str, torch.Tensor],
                                  anchors: List[torch.Tensor],
                                  ) -> Dict[str, torch.Tensor]:
        """
        Postprocess predictions for inference e.g. ocnvert logits to probs

        Args:
            Dict[str, torch.Tensor]: predictions from this head
                `box_logits`: classification logits for each anchor [N]
                `box_deltas`: offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            List[torch.Tensor]: anchors per image
        """
        postprocess_predictions = {
            "pred_boxes": self.coder.decode(prediction["box_deltas"], anchors),
            "pred_probs": self.classifier.box_logits_to_probs(prediction["box_logits"]),
        }
        return postprocess_predictions


class DetectionHeadHNM(DetectionHead):
    def __init__(self,
                 classifier: Classifier,
                 regressor: Regressor,
                 coder: BoxCoderND,
                 sampler: AbstractSampler,
                 log_num_anchors: Optional[str] = "mllogger",
                 ):
        """
        Detection head with classifier and regression module. Uses hard negative
        example mining to compute loss

        Args:
            classifier: classifier module
            regressor: regression module
            sampler (AbstractSampler): sampler for select positive and
                negative examples
            log_num_anchors (str): name of logger to use; if None, no logging
                will be performed
        """
        super().__init__(classifier=classifier, regressor=regressor, coder=coder)

        self.logger = None # get_logger(log_num_anchors) if log_num_anchors is not None else None
        self.fg_bg_sampler = sampler

    def compute_loss(self,
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
                box_logits (Tensor): classification logits for each anchor
                    [N, num_classes]
                box_deltas (Tensor): offsets for each anchor
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
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        target_labels = torch.cat(target_labels, dim=0)

        with torch.no_grad():
            batch_matched_gt_boxes = torch.cat(matched_gt_boxes, dim=0)
            batch_anchors = torch.cat(anchors, dim=0)
            target_deltas_sampled = self.coder.encode_single(
                batch_matched_gt_boxes[sampled_pos_inds], batch_anchors[sampled_pos_inds],
            )

        # target_deltas = self.coder.encode(matched_gt_boxes, anchors)
        # target_deltas_sampled = torch.cat(target_deltas, dim=0)[sampled_pos_inds]

        # assert len(batch_anchors) == len(batch_matched_gt_boxes)
        # assert len(batch_anchors) == len(box_deltas)
        # assert len(batch_anchors) == len(box_logits)
        # assert len(batch_anchors) == len(target_labels)

        if sampled_pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                box_deltas[sampled_pos_inds],
                target_deltas_sampled,
                ) / max(1, sampled_pos_inds.numel())

        losses["cls"] = self.classifier.compute_loss(
            box_logits[sampled_inds], target_labels[sampled_inds])
        return losses, sampled_pos_inds, sampled_neg_inds

    def select_indices(self,
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
        boxes_max_fg_probs = self.classifier.box_logits_to_probs(boxes_scores)
        boxes_max_fg_probs = boxes_max_fg_probs.max(dim=1)[0]  # search max of fg probs

        # positive and negative anchor indices per image
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(target_labels, boxes_max_fg_probs)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        # if self.logger:
        #     self.logger.add_scalar("train/num_pos", sampled_pos_inds.numel())
        #     self.logger.add_scalar("train/num_neg", sampled_neg_inds.numel())

        return sampled_pos_inds, sampled_neg_inds


class BoxHeadNoSampler(DetectionHead):
    def __init__(self,
                 classifier: Classifier,
                 regressor: Regressor,
                 coder: BoxCoderND,
                 log_num_anchors: Optional[str] = "mllogger",
                 **kwargs
                 ):
        """
        Detection head with classifier and regression module. Uses all
        foreground anchors for regression an passes all anchors to classifier

        Args:
            classifier: classifier module
            regressor: regression module
            log_num_anchors (str): name of logger to use; if None, no
                logging will be performed
        """
        super().__init__(classifier=classifier, regressor=regressor, coder=coder)
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
        return losses, sampled_pos_inds, None


class DetectionHeadHNMNative(DetectionHeadHNM):
    def compute_loss(self,
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
                box_logits (Tensor): classification logits for each anchor
                    [N, num_classes]
                box_deltas (Tensor): offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
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

        with torch.no_grad():
            losses = {}
            sampled_pos_inds, sampled_neg_inds = self.select_indices(target_labels, box_logits)
            sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        target_labels = torch.cat(target_labels, dim=0)
        batch_anchors = torch.cat(anchors, dim=0)
        pred_boxes_sampled = self.coder.decode_single(
            box_deltas[sampled_pos_inds], batch_anchors[sampled_pos_inds])

        target_boxes_sampled = torch.cat(matched_gt_boxes, dim=0)[sampled_pos_inds]

        if sampled_pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                pred_boxes_sampled,
                target_boxes_sampled,
                ) / max(1, sampled_pos_inds.numel())

        losses["cls"] = self.classifier.compute_loss(
            box_logits[sampled_inds], target_labels[sampled_inds])
        return losses, sampled_pos_inds, sampled_neg_inds


class DetectionHeadHNMNativeRegAll(DetectionHeadHNM):
    def compute_loss(self,
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
                box_logits (Tensor): classification logits for each anchor
                    [N, num_classes]
                box_deltas (Tensor): offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
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

        losses["cls"] = self.classifier.compute_loss(
            box_logits[sampled_inds], target_labels[sampled_inds])

        pos_inds = torch.where(target_labels >= 1)[0]
        pred_boxes = self.coder.decode_single(box_deltas[pos_inds], batch_anchors[pos_inds])
        target_boxes = torch.cat(matched_gt_boxes, dim=0)[pos_inds]

        if pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                pred_boxes,
                target_boxes,
                ) / max(1, pos_inds.numel())

        return losses, sampled_pos_inds, sampled_neg_inds


class DetectionHeadHNMRegAll(DetectionHeadHNM):
    def compute_loss(self,
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
                box_logits (Tensor): classification logits for each anchor
                    [N, num_classes]
                box_deltas (Tensor): offsets for each anchor
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
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        target_labels = torch.cat(target_labels, dim=0)

        losses["cls"] = self.classifier.compute_loss(
            box_logits[sampled_inds], target_labels[sampled_inds])

        pos_inds = torch.where(target_labels >= 1)[0]
        with torch.no_grad():
            batch_matched_gt_boxes = torch.cat(matched_gt_boxes, dim=0)
            batch_anchors = torch.cat(anchors, dim=0)
            target_deltas_sampled = self.coder.encode_single(
                batch_matched_gt_boxes[pos_inds], batch_anchors[pos_inds],
            )

        assert len(batch_anchors) == len(batch_matched_gt_boxes)
        assert len(batch_anchors) == len(box_deltas)
        assert len(batch_anchors) == len(box_logits)
        assert len(batch_anchors) == len(target_labels)

        if pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                box_deltas[pos_inds],
                target_deltas_sampled,
                ) / max(1, pos_inds.numel())

        return losses, sampled_pos_inds, sampled_neg_inds


class BoxHeadNativeEMA(DetectionHeadHNMNative):
    def __init__(self,
                 classifier: Classifier,
                 regressor: Regressor,
                 coder: BoxCoderND,
                 sampler: AbstractSampler,
                 log_num_anchors: Optional[str] = "mllogger",
                 beta: float = 0.9,
                 bias_correction: bool = True,
                 ):
        """
        Detection head with classifier and regression module. Uses hard negative
        example mining to compute loss

        Args:
            classifier: classifier module
            regressor: regression module
            sampler: sampler for select positive and negative examples
            log_num_anchors: name of logger to use; if None, no logging will be performed
            beta: beta parameter for exponential morving average
            bias_correction: use bias correction for exponential moving average
        
        Warnings:
            classifier loss is normalized in head -> use "sum" aggregation
            for classifier
        
        # WARNING: the current imeplementation does not normalize over the number of classes
        """
        super().__init__(classifier=classifier, regressor=regressor,
                         coder=coder, sampler=sampler, log_num_anchors=log_num_anchors)
        self.num_inds_cache = EMA(beta=beta, bias_correction=bias_correction)
        self.num_pos_inds_cache = EMA(beta=beta, bias_correction=bias_correction)

    def compute_loss(self,
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
                box_logits (Tensor): classification logits for each anchor
                    [N, num_classes]
                box_deltas (Tensor): offsets for each anchor
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
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        target_labels = torch.cat(target_labels, dim=0)
        batch_anchors = torch.cat(anchors, dim=0)
        pred_boxes_sampled = self.coder.decode_single(
            box_deltas[sampled_pos_inds], batch_anchors[sampled_pos_inds])

        target_boxes_sampled = torch.cat(matched_gt_boxes, dim=0)[sampled_pos_inds]

        assert len(batch_anchors) == len(box_deltas)
        assert len(batch_anchors) == len(box_logits)
        assert len(batch_anchors) == len(target_labels)

        if sampled_pos_inds.numel() > 0:
            self.num_pos_inds_cache.add(sampled_pos_inds.numel())
            losses["reg"] = self.regressor.compute_loss(
                pred_boxes_sampled,
                target_boxes_sampled,
                )
            losses["reg"] = losses["reg"] / max(1, self.num_pos_inds_cache.get())

        self.num_inds_cache.add(sampled_inds.numel())
        losses["cls"] = self.classifier.compute_loss(
            box_logits[sampled_inds], target_labels[sampled_inds])
        losses["cls"] = losses["cls"] / self.num_inds_cache.get()
        return losses, sampled_pos_inds, sampled_neg_inds


class EMA:
    def __init__(self, beta: float = 0.9, bias_correction: bool = True):
        """
        Exponentially weighted moving average
        new_cache = beta * cache + (1 - beta) * new_val
        Approximatley averages (1 - beta)^(-1) values

        Args:
            beta: weights for averaging
            bias_correction: applies bias correction
        """
        self.beta = beta
        self.cache: float = 0.

        self.bias_correction = bias_correction
        self.t = 0

    def add(self, val):
        """
        Add new value

        Args:
            val: new value to add
        """
        self.cache = self.beta * self.cache + (1 - self.beta) * float(val)
        self.t = min(self.t + 1, 100000)  # prevent overflow

    def get(self) -> float:
        """
        Retrive vurrent value

        Returns:
            float: current EMA
        """
        if self.bias_correction and self.t > 0:
            return (self.cache / (1 - pow(self.beta, self.t)))
        else:
            return self.cache


class BoxHNMNativeIoU(DetectionHeadHNMNative):
    def __init__(self,
                 classifier: Classifier,
                 regressor: Regressor,
                 coder: BoxCoderND,
                 sampler: AbstractSampler,
                 log_num_anchors: Optional[str] = "mllogger",
                 iou_cutoff: float = 1.0,
                 ):
        """
        Args:
            classifier: classifier module
            regressor: regression module
            sampler (AbstractSampler): sampler for select positive and
                negative examples
            log_num_anchors (str): name of logger to use; if None, no logging
                will be performed
            iou_cutoff: during inference the IoU wiill clipped to this
                value and rescaled to one.
        """
        super().__init__(
            classifier=classifier,
            regressor=regressor,
            coder=coder,
            sampler=sampler,
            log_num_anchors=log_num_anchors,
        )
        self.iou_cutoff = iou_cutoff

    def forward(self,
                fmaps: List[torch.Tensor],
                ) -> Dict[str, torch.Tensor]:
        """
        Forward feature maps through head modules

        Args:
            fmaps: list of feature maps for head module

        Returns:
            Dict[str, torch.Tensor]: predictions
                `box_deltas`(Tensor): bounding box offsets
                    [Num_Anchors_Batch, (dim * 2)]
                `box_logits`(Tensor): classification logits
                    [Num_Anchors_Batch, (num_classes)]
        """
        logits, offsets, ious_pred = [], [], []
        for level, p in enumerate(fmaps):
            logits.append(self.classifier(p, level=level))
            box_offsets, box_iou_predicted = self.regressor(p, level=level)
            offsets.append(box_offsets)
            ious_pred.append(box_iou_predicted)

        sdim = fmaps[0].ndim - 2
        box_deltas = torch.cat(offsets, dim=1).reshape(-1, sdim * 2)
        box_logits = torch.cat(logits, dim=1).flatten(0, -2)
        box_ious_pred = torch.cat(ious_pred, dim=1).flatten()

        return {"box_deltas": box_deltas,
                "box_logits": box_logits,
                "box_ious_pred": box_ious_pred,
                }

    def compute_loss(self,
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
                box_logits (Tensor): classification logits for each anchor
                    [N, num_classes]
                box_deltas (Tensor): offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
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
        box_ious_pred = prediction["box_ious_pred"]

        losses = {}
        sampled_pos_inds, sampled_neg_inds = self.select_indices(target_labels, box_logits)
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        target_labels = torch.cat(target_labels, dim=0)
        batch_anchors = torch.cat(anchors, dim=0)
        pred_boxes_sampled = self.coder.decode_single(
            box_deltas[sampled_pos_inds], batch_anchors[sampled_pos_inds])

        target_boxes_sampled = torch.cat(matched_gt_boxes, dim=0)[sampled_pos_inds]

        assert len(batch_anchors) == len(box_deltas)
        assert len(batch_anchors) == len(box_logits)
        assert len(batch_anchors) == len(target_labels)

        if sampled_pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                pred_boxes_sampled,
                target_boxes_sampled,
                box_ious_pred[sampled_pos_inds],
                ) / max(1, sampled_pos_inds.numel())

        losses["cls"] = self.classifier.compute_loss(
            box_logits[sampled_inds], target_labels[sampled_inds])
        return losses, sampled_pos_inds, sampled_neg_inds

    def postprocess_for_inference(self,
                                  prediction: Dict[str, torch.Tensor],
                                  anchors: List[torch.Tensor],
                                  ) -> Dict[str, torch.Tensor]:
        """
        Postprocess predictions for inference e.g. ocnvert logits to probs

        Args:
            Dict[str, torch.Tensor]: predictions from this head
                `box_logits`: classification logits for each anchor [N]
                `box_deltas`: offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            List[torch.Tensor]: anchors per image
        """
        probs = self.classifier.box_logits_to_probs(prediction["box_logits"])
        ious = torch.sigmoid(prediction["box_ious_pred"])
        ious = ious.clamp_(min=0, max=self.iou_cutoff) / self.iou_cutoff

        postprocess_predictions = {
            "pred_boxes": self.coder.decode(prediction["box_deltas"], anchors),
            "pred_probs": probs * ious[:, None],
        }
        return postprocess_predictions


HeadType = TypeVar('HeadType', bound=AbstractHead)
