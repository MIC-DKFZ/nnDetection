# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Original code from DETR https://github.com/facebookresearch/detr/blob/main/models/matcher.py
# SPDX-FileCopyrightText: 2020 Facebook
# SPDX-License-Identifier: Apache-2.0


from typing import Dict, List, Optional, Tuple, Union

import torch

from nndet.core.boxes import box_center_normalized_to_edges_original
from nndet.core.boxes.matcher1to1.base import BaseMatcher
from nndet.core.boxes.ops import (
    box_center2point_format,
    box_point2center_format,
    box_point_norm_with_size,
)
from nndet.nn.heads.classifier.ffn import FFNClassifier
from nndet.nn.heads.regressor.ffn import FFNRegressor
from nndet.utils.dist import get_world_size, is_dist_avail_and_initialized


class DETRHead(torch.nn.Module):
    def __init__(
        self,
        classifier: FFNClassifier,
        regressor: FFNRegressor,
        matcher: BaseMatcher,
        aux_loss: bool = True,
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self.reggressor = regressor
        self.matcher = matcher
        self.aux_loss = aux_loss

    def forward(
        self,
        out_sequence: torch.Tensor,
        reference: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[List[Dict[str, torch.Tensor]]]]:
        """
        Predict bounding boxes and classes using the ClassifierFFN and
        RegressionFFN

        Args:
            out_sequence: output sequence of the transformer [D, B, R, C]
                where D=number of decoder layers, B=batch size,
                R=number of predictions, C=number of channels
            reference: reference output of the transformer
                (not used in original DETR head)
                #TODO

        Returns:
            Dict[str, torch.Tensor]: predictions and auxiliary information

                ``"pred_cls_logits"`` torch.Tensor
                    predicted logits from ClassifierFFN [B, R, num_classes]
                    where B=batch size, R=number of predictions,
                    num_classes=number of classes

                ``"pred_box_coords"`` torch.Tensor
                    predicted normalized coords from RegressorFFN
                    [B, R, dims * 2] where B=batch size, R=number of
                    predictions, dims=number of spatial dimensions

                ``"aux_outputs"`` List[Dict[str, torch.Tensor]]
                    list with predictions from previous decoder layers
                    following the same format as `pred_cls_logits` and
                    `pred_box_coords`
        """
        box_logits = self.regressor(out_sequence)
        class_logits = self.classifier(out_sequence)

        preds = {"pred_cls_logits": class_logits[-1], "pred_box_coords": box_logits[-1]}
        # Predict for all decoder levels but only propagate last decoder output

        if self.aux_loss:
            # put remaining oututs into aux info
            aux = [{"pred_cls_logits": a, "pred_box_coords": b} for a, b in zip(class_logits[:-1], box_logits[:-1])]
            preds["aux_outputs"] = aux
        return preds

    def compute_loss(
        self,
        pred_detection: Dict[str, torch.Tensor],
        target_boxes: List[torch.Tensor],
        target_labels: List[torch.Tensor],
        img_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    ) -> Tuple[Dict[str, torch.Tensor]]:
        """
        Main function to compute losses (prepare targets, match and calculate)

        Args:
            pred_detection: predictions and auxiliary information

                ``"pred_cls_logits"`` torch.Tensor
                    predicted logits from ClassifierFFN [B, R, num_classes]
                    where B=batch size, R=number of predictions,
                    num_classes=number of classes

                ``"pred_box_coords"`` torch.Tensor
                    predicted normalized coords from RegressorFFN
                    [B, R, dims * 2] where B=batch size, R=number of
                    predictions, dims=number of spatial dimensions

                ``"aux_outputs"`` List[Dict[str, torch.Tensor]]
                    list with predictions from previous decoder layers
                    following the same format as `pred_cls_logits` and
                    `pred_box_coords`

            target_boxes: target boxes in point format List([N, dims * 2])
                (x0, y0, x1, y1 (,z0, z1))
            target_labels: target labels in numerical format List([N])
            img_shape: image size

        Returns:
            Dict[str, torch.Tensor]: computed losses
        """
        # average number of boxes for norm in distributed setting
        num_boxes = sum(len(t) if t.numel() > 0 else 0 for t in target_labels)
        num_boxes = torch.as_tensor(
            [num_boxes],
            dtype=torch.float,
            device=pred_detection["pred_cls_logits"].device,
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # change format of targets
        target_boxes, target_labels = self.prepare_targets(
            target_boxes=target_boxes,
            target_labels=target_labels,
            img_shape=img_shape,
        )

        # compute losses
        losses = self._match_and_compute_loss(
            pred_logits=pred_detection["pred_cls_logits"],
            pred_coords=pred_detection["pred_box_coords"],
            target_boxes=target_boxes,
            target_labels=target_labels,
            num_boxes=num_boxes,
        )
        if "aux_outputs" in pred_detection:
            for i, aux_outputs in enumerate(pred_detection["aux_outputs"]):
                l_dict = self._match_and_compute_loss(
                    pred_logits=pred_detection["pred_cls_logits"],
                    pred_coords=pred_detection["pred_box_coords"],
                    target_boxes=target_boxes,
                    target_labels=target_labels,
                    num_boxes=num_boxes,
                )
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                losses.update(l_dict)
                # TODO: check normalization
        return losses

    def prepare_targets(
        target_boxes: List[torch.Tensor],
        target_labels: List[torch.Tensor],
        img_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Prepare targets for loss computation

        Args:
            target_boxes: target boxes in point format List([N, dims * 2])
                (x0, y0, x1, y1 (,z0, z1))
            target_labels: target labels in numerical format List([N])
            img_shape: shape of image/patch

        Returns:
            torch.Tensor: target boxes in center format List([N, dims * 2])
                (cx, cy, dx, dy (,cz, dz))
            torch.Tensor: target labels in numerical format
        """
        target_boxes_new = []
        for box in target_boxes:
            boxes_norm = box_point_norm_with_size(box, img_shape=img_shape)
            target_boxes_new.append(box_point2center_format(boxes_norm))
        return target_boxes_new, target_labels

    def _match_and_compute_loss(
        self,
        pred_logits: torch.Tensor,
        pred_coords: torch.Tensor,
        target_boxes: List[torch.Tensor],
        target_labels: List[torch.Tensor],
        num_boxes: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform matching of predictions and ground truth objects and
        compute losses

        Args:
            pred_logits: predicted class logits by FFNClassifier
                [B, R, num_classes] where B=batch size, R=number of
                predictions, num_classes=number of classe
            pred_coords: predicted normalized coords from RegressorFFN
                [B, R, dims * 2] where B=batch size, R=number of
                predictions, dims=number of spatial dimensions
            target_boxes: target boxes in center format List([N, dims * 2])
                (cx, cy, dx, dy (,cz, dz)) (format referes to DETR default)
            target_labels: target labels in numerical format List([N])
            num_boxes: average number of ground truth boxes in the batch

        Returns:
            Dict[str, torch.Tensor]: computed losses. Exact entries depend on
                FFNClassifier and FFNRegressor
        """
        indices = self.matcher(
            pred_logits=pred_logits,
            pred_coords=pred_coords,
            target_boxes=target_boxes,
            target_labels=target_labels,
        )

        losses = {}
        losses.udpate(
            self.compute_class_loss(
                pred_logits=pred_logits,
                target_labels=target_labels,
                indices=indices,
                num_boxes=num_boxes,
            )
        )
        losses.udpate(
            self.compute_box_loss(
                pred_coords=pred_coords,
                target_boxes=target_boxes,
                indices=indices,
                num_boxes=num_boxes,
            )
        )
        return losses

    def compute_class_loss(
        self,
        pred_logits: torch.Tensor,
        target_labels: List[torch.Tensor],
        indices: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
        num_boxes: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute classification loss of head

        Args:
            pred_logits: predicted class logits by FFNClassifier
                [B, R, num_classes] where B=batch size, R=number of
                predictions, num_classes=number of classe
            target_labels: target labels in numerical format List([N])
            indices: indices to match predictions and ground truth objects
                as obtained from matcher class (empty images encoded
                as None entries)
            num_boxes: average number of ground truth boxes in the batch

        Returns:
            Dict[str, torch.Tensor]: computed classification losses
        """
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([target_labels[J] for t, (_, J) in zip(target_labels, indices) if J is not None])
        target_classes = torch.full(
            pred_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=pred_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss = self.classifier.compute_loss(
            pred_logits=pred_logits.transpose(1, 2),
            targets=target_classes,
        )
        # TODO: add class error
        return loss

    def compute_box_loss(
        self,
        pred_coords: torch.Tensor,
        target_boxes: List[torch.Tensor],
        indices: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
        num_boxes_all: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute regression loss of head

        Args:
            pred_coords: predicted normalized coords from RegressorFFN
                [B, R, dims * 2] where B=batch size, R=number of
                predictions, dims=number of spatial dimensions
            target_boxes: target boxes in center format List([N, dims * 2])
                (cx, cy, dx, dy (,cz, dz)) (format referes to DETR default)
            indices: indices to match predictions and ground truth objects
                as obtained from matcher class (empty images encoded
                as None entries)
            num_boxes: average number of ground truth boxes in the batch

        Returns:
            Dict[str, torch.Tensor]: computed regression losses
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = pred_coords[idx]

        target_boxes = torch.cat(
            [target_boxes[i] for t, (_, i) in zip(target_boxes, indices) if i is not None],
            dim=0,
        )

        loss = self.reggressor.compute_loss(
            preds=src_boxes,
            targets=target_boxes,
            pred_boxes=box_center2point_format(src_boxes),
            target_boxes=box_center2point_format(target_boxes),
        )
        loss = {key: item / num_boxes_all for key, item in loss.items()}
        return loss

    @staticmethod
    def _get_src_permutation_idx(
        indices: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Permute predictions following indices

        Args:
            indices: paried indices as obtained from `matcher1to1`

        Returns:
            torch.Tensor: tensor containing the batch indices of the
                predictions to bring them into same order as ground truth
            torch.Tensor: tensor cotaining the element indices of
                the predictions to bring them into the same order as the
                ground truth
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices) if src is not None])
        src_idx = torch.cat([src for (src, _) in indices if src is not None])
        return batch_idx, src_idx

    def postprocess_for_inference(
        self,
        pred_detection: Dict[str, torch.Tensor],
        img_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Postprocessing of the Predictions

        Args:
            pred_detection: prediction from network

                ``"pred_cls_logits"`` torch.Tensor
                    predicted logits from ClassifierFFN [B, R, num_classes]
                    where B=batch size, R=number of predictions,
                    num_classes=number of classes

                ``"pred_box_coords"`` torch.Tensor
                    predicted normalized coords from RegressorFFN
                    [B, R, dims * 2] where B=batch size, R=number of
                    predictions, dims=number of spatial dimensions

            img_shape: image size

        Returns:
                Dict: post processed predictions

                    ``'pred_boxes'``: List[torch.ensor]
                        predicted bounding boxes for each image List[[R, dim * 2]]

                    ``'pred_scores'``: List[torch.Tensor]
                        predicted probability for the class List[[R]]

                    ``'pred_labels'``: List[torch.Tensor]
                        predicted class List[[R]]
        """
        batch_pred_scores_fg = self.classifier.postprocess_logits(pred_detection["pred_cls_logits"])
        batch_pred_scores, batch_pred_labels = batch_pred_scores_fg.max(-1)
        # TODO: replace deprecated function
        batch_pred_boxes = box_center_normalized_to_edges_original(pred_detection["pred_box_coords"], img_shape)

        batch_size = batch_pred_scores.shape[0]
        assert batch_size == batch_pred_labels.shape[0]
        assert batch_size == batch_pred_boxes.shape[0]
        assert batch_pred_labels.shape[1] == batch_pred_boxes.shape[1]

        prediction = {"pred_boxes": [], "pred_scores": [], "pred_labels": []}
        for batch_idx in range(batch_size):
            prediction["pred_boxes"].append(batch_pred_boxes[batch_idx])
            prediction["pred_scores"].append(batch_pred_scores[batch_idx])
            prediction["pred_labels"].append(batch_pred_labels[batch_idx])
        return prediction
