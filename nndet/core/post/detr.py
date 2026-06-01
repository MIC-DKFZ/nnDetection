# Modifications licensed under:
# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# MaxFGBoxPost adapted from DETR https://github.com/facebookresearch/detr
# SPDX-FileCopyrightText: 2020 Facebook, Inc
# SPDX-License-Identifier: Apache-2.0
#
# TopKBoxPost adapted from Deformable-DETR https://github.com/fundamentalvision/Deformable-DETR
# SPDX-FileCopyrightText: 2020 SenseTime
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch


class DETRBoxPost:
    def process_batch(
        self,
        pred_scores: torch.Tensor,
        pred_boxes: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Postprocessing of the Predictions

        Args:
            pred_scores: predicted probabilities for a batch of boxes
                [B, R, num_classes] where B=batch size, R=number of predictions,
                num_classes=number of classes
            pred_boxes: predicted batch of boxes [B, R, dims * 2]
                where B=batch size, R=number of predictions, dims=number of
                spatial dimensions

        Returns:
            pred_boxes: predicted bounding boxes for each image
                List[[R, dim * 2]]
            pred_scores: predicted probability for the class List[[R]]
            pred_labels: predicted class List[[R]]
        """
        raise NotImplementedError


class MaxFGBoxPost(DETRBoxPost):
    def process_batch(
        self,
        pred_scores: torch.Tensor,
        pred_boxes: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Postprocessing of the Predictions. Selects the maximal foreground
        probability for each prediction. Usually used in combination with
        softmax based losses.

        Args:
            pred_scores: predicted probabilities for a batch of boxes
                [B, R, num_classes] where B=batch size, R=number of predictions,
                num_classes=number of classes
            pred_boxes: predicted batch of boxes [B, R, dims * 2]
                where B=batch size, R=number of predictions, dims=number of
                spatial dimensions

        Returns:
            pred_boxes: predicted bounding boxes for each image
                List[[R, dim * 2]]
            pred_scores: predicted probability for the class List[[R]]
            pred_labels: predicted class List[[R]]
        """
        pred_scores, pred_labels = pred_scores.max(-1)  # [B, R]

        # unpack into lists and perform sanity checks
        batch_size = pred_scores.shape[0]
        assert batch_size == pred_labels.shape[0]
        assert batch_size == pred_boxes.shape[0]
        assert pred_labels.shape[1] == pred_boxes.shape[1]
        assert pred_labels.shape[1] == pred_boxes.shape[1]
        return (
            [p[0] for p in pred_boxes.split(1, dim=0)],
            [p[0] for p in pred_scores.split(1, dim=0)],
            [p[0] for p in pred_labels.split(1, dim=0)],
        )


class TopKBoxPost(DETRBoxPost):
    def __init__(self, topk: int) -> None:
        """
        Postprcoessing strategy during inference. Selects the topk predictions
        (irrespective of their class) for detections. Usually used in
        combination with sigmoid based losses.

        Args:
            topk: number of predictions to select.
        """
        super().__init__()
        self.topk = topk

    def process_batch(
        self,
        pred_scores: torch.Tensor,
        pred_boxes: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Postprocessing of the Predictions. Selects the topk predictions
        (irrespective of their class) for detections.

        Args:
            pred_scores: predicted probabilities for a batch of boxes
                [B, R, num_classes] where B=batch size, R=number of predictions,
                num_classes=number of classes
            pred_boxes: predicted batch of boxes [B, R, dims * 2]
                where B=batch size, R=number of predictions, dims=number of
                spatial dimensions

        Returns:
            pred_boxes: predicted bounding boxes for each image
                List[[R, dim * 2]]
            pred_scores: predicted probability for the class List[[R]]
            pred_labels: predicted class List[[R]]
        """
        dim = pred_boxes.shape[-1] // 2

        # flatten query and class dimension
        pred_scores_topk, topk_indices = torch.topk(pred_scores.view(pred_scores.shape[0], -1), self.topk, dim=1)
        # index div classes gives the to the index corresponding query
        topk_boxes = torch.div(topk_indices, pred_scores.shape[2], rounding_mode="floor")
        pred_labels_topk = topk_indices % pred_scores.shape[2]
        pred_boxes_topk = torch.gather(pred_boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, dim * 2))

        # unpack into lists and perform sanity checks
        batch_size = pred_scores_topk.shape[0]
        assert batch_size == pred_labels_topk.shape[0]
        assert batch_size == pred_boxes_topk.shape[0]
        assert pred_labels_topk.shape[1] == pred_boxes_topk.shape[1]
        assert pred_scores_topk.shape[1] == pred_boxes_topk.shape[1]
        return (
            [p[0] for p in pred_boxes_topk.split(1, dim=0)],
            [p[0] for p in pred_scores_topk.split(1, dim=0)],
            [p[0] for p in pred_labels_topk.split(1, dim=0)],
        )
