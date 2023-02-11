# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch

from nndet.core.boxes import BoxCoderND
from nndet.nn.heads.abstract import BaseHead
from nndet.nn.heads.classifier.dense import DenseClassifier
from nndet.nn.heads.classifier.roi import RoIClassifier
from nndet.nn.heads.regressor.dense import DenseRegressor
from nndet.nn.heads.regressor.roi import RoIRegressor
from nndet.utils.enums import BoxRegressionMode


class AnchorHead(BaseHead):
    def __init__(
        self,
        classifier: DenseClassifier,
        regressor: DenseRegressor,
        coder: BoxCoderND,
        shared: Optional[torch.nn.Module] = None,
        reg_mode: Union[str, BoxRegressionMode] = "decode",
    ):
        """
        Provides an abstract interface for an module which takes
        inputs and computed its own loss

        Args:
            classifier: classifier module
            regressor: regression module
            coder: Module to encoder/decoder box delta wrt to anchors/proposals
            shared: optional shared module which is applied to before the
                classifier and regression head
            reg_mode: define regression mode. One of `decode` | `encode`

                ``'decode'``
                    uses the predicted box deltas to decode the
                    predicted boxes which are passed to the regression loss
                    in combination with the matched ground truth boxes

                ``'encode'``
                    uses the matched ground truth to encode the
                    expected box deltas which are passed to the regression loss
                    in combination with the predicted box deltas

        """
        super().__init__(
            classifier=classifier,
            regressor=regressor,
            shared=shared,
            coder=coder,
        )
        self.reg_mode = BoxRegressionMode(reg_mode)

    def forward(
        self,
        fmaps: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward feature maps through head modules

        Args:
            fmaps: list of feature maps for head module
                [N, C, spatial_dims]

        Returns:
            Dict[str, torch.Tensor]: predictions

                ``'box_deltas'`` (Tensor)
                    bounding box offsets
                    [Num_Anchors_Batch, (num_classes), dim * 2];
                    num classes is only present if anchors were regressed
                    for each class individually

                ``'box_logits'`` (Tensor)
                    classification logits
                    [Num_Anchors_Batch, num_classes]

        """
        logits, offsets = [], []
        for level, p in enumerate(fmaps):
            if self.shared is not None:
                intermediate = self.shared(p, level=level)
            else:
                intermediate = p

            offsets.append(self.regressor(intermediate, level=level))
            logits.append(self.classifier(intermediate, level=level))

        sdim = fmaps[0].ndim - 2
        if self.is_class_agnostic():
            box_deltas = torch.cat(offsets, dim=1).reshape(-1, sdim * 2)
        else:
            # TODO multi class regression
            raise NotImplementedError
        box_logits = torch.cat(logits, dim=1).flatten(0, -2)
        return {"box_deltas": box_deltas, "box_logits": box_logits}

    def postprocess_for_inference(
        self,
        prediction: Dict[str, torch.Tensor],
        anchors: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Postprocess predictions for inference e.g. ocnvert logits to probs

        Args:
            Dict[str, torch.Tensor]: predictions from this head

                ``'box_logits'``
                    classification logits for each anchor [N]

                ``'box_deltas'``
                    offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]

            List[torch.Tensor]: anchors per image
        """
        postprocess_predictions = {
            "pred_boxes": self.coder.decode(
                prediction["box_deltas"],
                anchors,
            ),
            "pred_probs": self.classifier.logits_to_probs(prediction["box_logits"]),
        }
        return postprocess_predictions

    def get_reg_by_mode(
        self,
        batch_anchors: torch.Tensor,
        batch_target_boxes: torch.Tensor,
        batch_pred_deltas: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute regression targets

        Args:
            batch_anchors: concatenated anchors
            batch_target_boxes: concatenated matched ground truth box
            batch_pred_deltas: concatenated predicted box deltas

        Returns:
            Tuple[Tensor, Tensor]: (predicted regression values,
                expected regression values)
                `encode`: predicted box deltas, target box deltas
                `decode`: predicted boxes, target boxes
        """
        if self.reg_mode == BoxRegressionMode.ENCODE:
            target_deltas = self.coder.encode_single(batch_target_boxes, batch_anchors)
            return batch_pred_deltas, target_deltas
        elif self.reg_mode == BoxRegressionMode.DECODE:
            pred_boxes = self.coder.decode_single(batch_pred_deltas, batch_anchors)
            return pred_boxes, batch_target_boxes
        else:
            raise RuntimeError("Wrong mode.")

    @abstractmethod
    def compute_loss(
        self,
        prediction: Dict[str, torch.Tensor],
        target_labels: List[torch.Tensor],
        matched_gt_boxes: List[torch.Tensor],
        anchors: List[torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Compute regression and classification loss
        N anchors over all images; M anchors per image => sum(M) = N

        Args:
            prediction: detection predictions for loss computation

                ``'box_logits'``
                    classification logits for each anchor
                    [N, num_classes]

                ``'box_deltas'``
                    offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]

            target_labels: target labels for each anchor (per image) [M]
            matched_gt_boxes: matched gt box for each anchor
                List[[M, dim *  2]]
            anchors: anchors per image List[[M, dim *  2]]

        Returns:
            Tensor: dict with losses (reg for regression loss, cls for
                classification loss)
            Tensor: sampled positive indices of anchors
                (after concatenation if sampled otherwise None)
            Tensor: sampled negative indices of anchors
            (after concatenation, if sampled otherwise None)
        """
        raise NotImplementedError


class RoIHead(BaseHead):
    def __init__(
        self,
        classifier: RoIClassifier,
        regressor: RoIRegressor,
        coder: BoxCoderND,
        shared: Optional[torch.nn.Module] = None,
        reg_mode: Union[str, BoxRegressionMode] = "encode",  # TODO: move this to a regressor function
    ):
        """
        Provides an abstract interface for an module which takes
        inputs and computed its own loss

        Args:
            classifier: classifier module
            regressor: regression module
            coder: Module to encoder/decoder box delta wrt to anchors/proposals
            shared: optional shared module which is applied to before the
                classifier and regression head
            reg_mode: define regression mode. One of `decode` | `encode`

                ``'decode'``
                    uses the predicted box deltas to decode the
                    predicted boxes which are passed to the regression loss
                    in combination with the matched ground truth boxes

                ``'encode'``
                    uses the matched ground truth to encode the
                    expected box deltas which are passed to the regression loss
                    in combination with the predicted box deltas

        """
        super().__init__(
            classifier=classifier,
            regressor=regressor,
            shared=shared,
            coder=coder,
        )
        self.reg_mode = BoxRegressionMode(reg_mode)

    def forward(
        self,
        fmaps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward feature maps through RoI head modules

        Args:
            fmaps: feature maps exracted from pooling operation for each
                proposal [num_proposals, C, spatial_dims]

        Returns:
            Dict[str, torch.Tensor]: predictions

                ``'box_deltas'`` (Tensor)
                    bounding box offsets
                    [num_proposals, (num_classes), dim * 2];
                    num classes is only present if anchors were regressed
                    for each class individually

                ``'box_logits'`` (Tensor)
                    classification logits [num_proposals, num_classes]
        """
        if self.shared is not None:
            intermediate = self.shared(fmaps)
        else:
            intermediate = fmaps

        box_deltas = self.regressor(intermediate)
        box_logits = self.classifier(intermediate)

        # TODO: check if reshape is needed

        return {
            "box_deltas": box_deltas,
            "box_logits": box_logits,
        }

    def postprocess_for_inference(
        self,
        prediction: Dict[str, torch.Tensor],
        anchors: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Postprocess predictions for inference e.g. ocnvert logits to probs

        Args:
            Dict[str, torch.Tensor]: predictions from this head

                ``'box_logits'``
                    classification logits for each anchor [N]

                ``'box_deltas'``
                    offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]

            List[torch.Tensor]: anchors per image
        """
        # TODO: check if anchors is a list of boxes for rois -> decode single
        postprocess_predictions = {
            "pred_boxes": self.coder.decode(
                prediction["box_deltas"],
                anchors,
            ),
            "pred_probs": self.classifier.logits_to_probs(prediction["box_logits"]),
        }
        return postprocess_predictions

    def get_reg_by_mode(
        self,
        batch_anchors: torch.Tensor,
        batch_target_boxes: torch.Tensor,
        batch_pred_deltas: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute regression targets

        Args:
            batch_anchors: concatenated anchors
            batch_target_boxes: concatenated matched ground truth box
            batch_pred_deltas: concatenated predicted box deltas

        Returns:
            Tuple[Tensor, Tensor]: (predicted regression values,
                expected regression values)
                `encode`: predicted box deltas, target box deltas
                `decode`: predicted boxes, target boxes
        """
        if self.reg_mode == BoxRegressionMode.ENCODE:
            target_deltas = self.coder.encode_single(batch_target_boxes, batch_anchors)
            return batch_pred_deltas, target_deltas
        elif self.reg_mode == BoxRegressionMode.DECODE:
            pred_boxes = self.coder.decode_single(batch_pred_deltas, batch_anchors)
            return pred_boxes, batch_target_boxes
        else:
            raise RuntimeError("Wrong mode.")

    @abstractmethod
    def compute_loss(
        self,
        prediction: Dict[str, torch.Tensor],
        target_labels: torch.Tensor,
        matched_gt_boxes: torch.Tensor,
        proposal_boxes: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Compute regression and classification loss

        Args:
            prediction: detection predictions for loss computation

                ``'box_logits'``
                    classification logits for each proposal
                    [N, num_classes]

                ``'box_deltas'``
                    offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]

            target_labels: target labels for each proposal [N]
            matched_gt_boxes: matched gt box for each proposal
                [N, dim *  2]
            proposal_boxes: concatenated and extended proposals with batch index
                (batch_idx, x1, y1, x2, y2, (z1, z2))[N, 1 + dim * 2]

        Returns:
            Tensor: dict with losses (reg for regression loss, cls for
                classification loss)
        """
        raise NotImplementedError
