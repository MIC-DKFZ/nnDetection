# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

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
        """
        super().__init__(
            classifier=classifier,
            regressor=regressor,
            shared=shared,
            coder=coder,
        )

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

                ``'box_deltas'`` (torch.Tensor)
                    bounding box offsets
                    [Num_Anchors_Batch, (num_classes), dim * 2];
                    num classes is only present if anchors were regressed
                    for each class individually

                ``'box_logits'`` (torch.Tensor)
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
            raise NotImplementedError("Dense/anchor regressors are currently limited to class agnostic regression.")
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

                ``'box_logits'`` (torch.Tensor)
                    classification logits for each anchor [N]

                ``'box_deltas'`` (torch.Tensor)
                    offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]

            List[torch.Tensor]: anchors per image

        # TODO: return
        # TODO: tests
        # TODO: docs
        """
        postprocess_predictions = {
            "pred_boxes": self.coder.decode(prediction["box_deltas"], anchors),
            "pred_probs": self.classifier.logits_to_probs(prediction["box_logits"]),
        }
        return postprocess_predictions

    def get_reg_targets_by_mode(
        self,
        batch_anchors: torch.Tensor,
        batch_target_boxes: torch.Tensor,
        batch_pred_deltas: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute regression targets

        Args:
            batch_anchors: concatenated anchors/proposals
            batch_target_boxes: concatenated matched ground truth box
            batch_pred_deltas: concatenated predicted box deltas

        Returns:
            Tuple[Tensor, Tensor]: (predicted regression values,
                expected regression values)
                `encode`: predicted box deltas, target box deltas
                `decode`: predicted boxes, target boxes
        """
        if self.regressor.get_reg_mode() == BoxRegressionMode.ENCODE:
            target_deltas = self.coder.encode_single(batch_target_boxes, batch_anchors)
            return batch_pred_deltas, target_deltas
        elif self.regressor.get_reg_mode() == BoxRegressionMode.DECODE:
            pred_boxes = self.coder.decode_single(batch_pred_deltas, batch_anchors)
            return pred_boxes, batch_target_boxes
        else:
            raise ValueError(
                f"Provided regressor {self.regressor.__class__.__name__} "
                f"with reg mode {self.regressor.get_reg_mode()} is not compatible "
                f"with {self.__class__.__name__}"
            )

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

                ``'box_logits'`` (torch.Tensor)
                    classification logits for each anchor
                    [N, num_classes]

                ``'box_deltas'`` (torch.Tensor)
                    offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]

            # TODO: rename target labels
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
    ):
        """
        Provide a base class for heads which can process RoIs. Provides
        implementations to forward through subnetworks, postprocess
        predictions and retrieve targets. The loss computation
        depends on the subclasses.

        Args:
            classifier: classifier module
            regressor: regression module
            coder: Module to encoder/decoder box delta wrt to anchors/proposals
            shared: optional shared module which is applied to before the
                classifier and regression head
        """
        super().__init__(
            classifier=classifier,
            regressor=regressor,
            shared=shared,
            coder=coder,
        )

    def forward(
        self,
        fmaps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward feature maps through RoI head modules

        Args:
            fmaps: feature maps exracted from pooling operation for each
                proposal [N, C, dims], where N is the number of RoIs,
                C is the number of input channels and dims are
                spatial dimensions

        Returns:
            Dict[str, torch.Tensor]: predictions

                ``'box_deltas'`` torch.Tensor
                    bounding box deltas of shape [N, (num_classes *) dim * 2],
                    where N=number of RoIs, dim=number of spatial dimensions,
                    and num_classes is the number of foreground classes.
                    num_classes is only used for class specific regression.

                ``'box_logits'`` torch.Tensor
                    classification logits [N, num_classes] where N is the
                    number of RoIs and num_classes is the number of foreground
                    classes
        """
        if self.shared is not None:
            intermediate = self.shared(fmaps)
        else:
            intermediate = fmaps

        box_deltas = self.regressor(intermediate)
        box_logits = self.classifier(intermediate)

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

                ``'box_deltas'`` torch.Tensor
                    bounding box deltas of shape [N, (num_classes *) dim * 2],
                    where N=number of RoIs, dim=number of spatial dimensions,
                    and num_classes is the number of foreground classes.
                    num_classes is only used for class specific regression.

                ``'box_logits'`` torch.Tensor
                    classification logits [N, num_classes] where N is the
                    number of RoIs and num_classes is the number of foreground
                    classes

            List[torch.Tensor]: anchor / proposals for each image of shape
                [N, dim * 2] where N is the number of RoIs per image,
                and dim is the number of spatial dimensions

        Returns:
            Dict[str, Tensor]: postprocessed predictions

                ``'pred_boxes'`` torch.Tensor
                    predicted bounding boxes in
                    (x1, y1, x2, y2, (z1, z2) (* num_classes)) format with
                    shape [N, (num_classes *) dim * 2], where N is the number
                    of RoIs and dim is the number of spatial dimensions.
                    num_classes is the number of foreground classes
                    and only present if class specific regression is used.

                ``'pred_probs'`` torch.Tensor
                    predicted probabilities/scores [N, num_classes], where
                    N is the number of RoIs and num_classes is the number
                    of foreground classes
        """
        postprocess_predictions = {
            "pred_boxes": self.coder.decode(prediction["box_deltas"], anchors),
            "pred_probs": self.classifier.logits_to_probs(prediction["box_logits"]),
        }
        return postprocess_predictions

    def get_reg_targets_by_mode(
        self,
        batch_anchors: torch.Tensor,
        batch_target_boxes: torch.Tensor,
        batch_pred_deltas: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute regression targets

        Args:
            batch_anchors: concatenated anchors/proposals
            batch_target_boxes: concatenated matched ground truth box
            batch_pred_deltas: concatenated predicted box deltas

        Returns:
            Tuple[Tensor, Tensor]: (predicted regression values,
                expected regression values)
                `encode`: predicted box deltas, target box deltas
                `decode`: predicted boxes, target boxes
        """
        if self.regressor.get_reg_mode() == BoxRegressionMode.ENCODE:
            target_deltas = self.coder.encode_single(batch_target_boxes, batch_anchors)
            return batch_pred_deltas, target_deltas
        elif self.regressor.get_reg_mode() == BoxRegressionMode.DECODE:
            pred_boxes = self.coder.decode_single(batch_pred_deltas, batch_anchors)
            return pred_boxes, batch_target_boxes
        else:
            raise ValueError(
                f"Provided regressor {self.regressor.__class__.__name__} "
                f"with reg mode {self.regressor.get_reg_mode()} is not compatible "
                f"with {self.__class__.__name__}"
            )

    @abstractmethod
    def compute_loss(
        self,
        prediction: Dict[str, torch.Tensor],
        matched_gt_labels: torch.Tensor,
        matched_gt_boxes: torch.Tensor,
        proposal_boxes: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
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
                N is the number of RoIs
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
        raise NotImplementedError
