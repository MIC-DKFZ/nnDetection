# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Tuple

import torch

from nndet.core.abstract import AbstractDetector, AbstractOneStageDetector
from nndet.core.rois.module import BaseRoIModule


class RCNN(AbstractDetector):
    def __init__(
        self,
        rpn: AbstractOneStageDetector,
        roi_module: BaseRoIModule,
    ) -> None:
        """
        Two stage detection module

        Args:
            rpn: one stage detector which generates an initial set of proposals
            roi_module: module which takes the proposals and generates final
                predictions
        """
        super().__init__()
        self.rpn = rpn
        self.roi_module = roi_module

    def train_step(
        self,
        images: torch.Tensor,
        targets: dict,
        batch_num: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a single training step (forward pass + loss computation)

        Args:
            images: batch of images
            targets: labels for training

                ``"target_boxes"`` (List[Tensor])
                    ground truth bounding boxes  (x1, y1, x2, y2, (z1, z2))
                    [X, dim * 2], X= number of  ground truth boxes in image

                ``"target_classes"`` (List[Tensor])
                    ground truth class per box (classes start from 0) [X],
                    X= number of ground truth boxes in image

                ``"target_binary_masks"`` List[Tensor]
                    Only required when additional mask head is provided.
                    associated binary mask for each ground truth object
                    [R, image_size]. The i-th entry along the first dimension
                    corresponds to the i-th object / box / class.

                ``"target_seg"`` (Tensor)
                    segmentation ground truth (only needed if ::param::`segmenter`
                    was provided in init) (classes start from 1, 0 background)

            batch_num: batch index inside epoch

        Returns:
            Dict: all losses
        """
        # copy target classes
        targets["target_roi_classes"] = [trc.detach().clone() for trc in targets["target_classes"]]
        # map original targets to fg vs bg for RPN
        targets["target_classes"] = [torch.zeros_like(trc) for trc in targets["target_classes"]]

        losses, proposals, features = self.rpn.train_step_with_features(
            images=images,
            targets=targets,
            predict=True,
            batch_num=batch_num,
        )

        targets.pop("target_classes")  # remove targets to avoid accidental class mixup

        roi_losses = self.roi_module.train_step(
            images=images,
            features=features,
            proposals=proposals,
            targets=targets,
        )

        losses.update(roi_losses)
        return losses

    @torch.no_grad()
    def validation_step(
        self,
        images: torch.Tensor,
        targets: dict,
        batch_num: bool,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Perform a single validation step (forward pass + loss computation)

        Args:
            images: batch of images
            targets: labels for training

                ``"target_boxes"`` (List[Tensor])
                    ground truth bounding boxes  (x1, y1, x2, y2, (z1, z2))
                    [X, dim * 2], X= number of  ground truth boxes in image

                ``"target_classes"`` (List[Tensor])
                    ground truth class per box (classes start from 0) [X],
                    X= number of ground truth boxes in image

                ``"target_binary_masks"`` List[Tensor]
                    Only required when additional mask head is provided.
                    associated binary mask for each ground truth object
                    [R, image_size]. The i-th entry along the first dimension
                    corresponds to the i-th object / box / class.

                ``"target_seg"`` (Tensor)
                    segmentation ground truth (only needed if ::param::`segmenter`
                    was provided in init) (classes start from 1, 0 background)

            batch_num: batch index inside epoch

        Returns:
            Dict: usually losses would be placed here but since RoI head is
                trained with HNM it is to expensive to compute losses,
                so this will always return a 'placeholder' with value '0'
            Dict[str, Any]: predictions

                ``"pred_boxes"`` List[Tensor]
                    predicted boxes [N, dims * 2]
                    (x_min, y_min, x_max, y_max, z_min, z_max)

                ``"pred_scores"`` List[Tensor]
                    associated scores for each predicted box [N]

                ``"pred_labels"`` List[Tensor]
                    associated labels for each predicted box [N]

                ``"pred_masks"`` List[Tensor]
                    predicted probability masks from mask head [N, RoI_dims]
                    The output size of the masks are determined by the Masker
                    RoI Module. To compute the evaluated additional post-
                    processing will be required.

                ``"pred_mask_scores"`` List[Tensor]
                    associated scores for each predicted masks [N]

                ``"pred_mask_labels"`` List[Tensor]
                    associated labels for each predicted masks [N]

                ``"pred_image_spatial_size"`` ND_TUPLE_INT
                    image size which was used for prediction. Needed to restore
                    correct size of image when pasting binary masks.
        """
        predictions = self.inference_step(images=images)
        losses = {"placeholder": torch.tensor(0)}
        return losses, predictions

    @torch.no_grad()
    def inference_step(
        self,
        images: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform a single inference step (forward pass)

        Args:
            images: batch of images

        Returns:
            Dict[str, Any]: predictions

                ``"pred_boxes"`` List[Tensor]
                    predicted boxes [N, dims * 2]
                    (x_min, y_min, x_max, y_max, z_min, z_max)

                ``"pred_scores"`` List[Tensor]
                    associated scores for each predicted box [N]

                ``"pred_labels"`` List[Tensor]
                    associated labels for each predicted box [N]

                ``"pred_masks"`` List[Tensor]
                    predicted probability masks from mask head [N, RoI_dims]
                    The output size of the masks are determined by the Masker
                    RoI Module. To compute the evaluated additional post-
                    processing will be required.

                ``"pred_mask_scores"`` List[Tensor]
                    associated scores for each predicted masks [N]

                ``"pred_mask_labels"`` List[Tensor]
                    associated labels for each predicted masks [N]

                ``"pred_image_spatial_size"`` ND_TUPLE_INT
                    image size which was used for prediction. Needed to restore
                    correct size of image when pasting binary masks.
        """
        proposals, features = self.rpn.inference_step_with_features(images=images, **kwargs)
        predictions = {f"rpn_{key}": item for key, item in proposals.items()}

        roi_predictions = self.roi_module.inference_step(
            images=images,
            features=features,
            proposals=proposals,
        )
        predictions.update(roi_predictions)
        return predictions
