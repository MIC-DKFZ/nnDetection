# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from nndet.core.abstract import AbstractDetector
from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.heads.detr.base import DETRHead
from nndet.nn.heads.segmenter import Segmenter
from nndet.nn.layers.pos_embed.sine import BasePositionEmbedding
from nndet.nn.neck.channel_mapper import ChannelMapper
from nndet.nn.transformer.abstract_transformer import AbstractTransformer


class BaseDETR(AbstractDetector):
    def __init__(
        self,
        backbone: AbstractBackbone,
        channel_mapper: ChannelMapper,
        transformer: AbstractTransformer,
        head: DETRHead,
        pos_embed: BasePositionEmbedding,
        hidden_dim: int,
        detection_per_img: int,
        query_dim: int,
        segmenter: Optional[Segmenter] = None,
        two_stage: bool = False,
        use_pos_queries: bool = False,  # TODO: docs
    ):
        """
        Basic DETR Module, Implements forward pass, loss computation

        Args:
            backbone: Backbone network to compute image features
            channel_mapper: Module that maps the features channel dimension
                to the hidden_dim in the transformer
            transformer: Transformer Model
            head: Head used for classification, regression, loss computation and
                postprocessing
            pos_embed: module to generate positional embedding
            hidden_dim: Dimension of the transformer sequence
            detection_per_img: number of detections the model does per patch
            pos_embed: module to generate positional embedding
            query_dim: dimension of object queries in the decoder
            segmenter: (Optional) segmenter to predict a semantic segmentations
                from the feature maps
            two_stage: toggle whether the encoder should predict objects and use
                those as query candidates

        """
        super().__init__()
        # Obtain important hyperparameters
        self.detection_per_img = detection_per_img
        # Set Backbone and get channels and feature levels
        self.backbone = backbone
        self.channel_mapper = channel_mapper
        self.hidden_dim = hidden_dim
        self.two_stage = two_stage

        # Build Transformer Specific Architecture
        self.pos_embed = pos_embed
        self.transformer = transformer
        if use_pos_queries:
            query_dim = 2 * query_dim
        self.query_pos = nn.Embedding(detection_per_img, query_dim) if not two_stage else None

        # Build the final layers for classification and box regression
        self.head = head

        # Build optional modules
        self.segmenter = segmenter

    def train_step(
        self,
        images: Tensor,
        targets: dict,
        batch_num: int,
    ) -> Dict[str, torch.Tensor]:
        """
        See `self.train_step_with_features` for more info
        """
        losses, _, _ = self.train_step_with_features(
            images=images,
            targets=targets,
            predict=False,
            batch_num=batch_num,
        )
        return losses

    @torch.no_grad()
    def validation_step(
        self,
        images: Tensor,
        targets: dict,
        batch_num: bool,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        See `self.train_step_with_features` for more info
        """
        losses, prediction, _ = self.train_step_with_features(
            images=images,
            targets=targets,
            predict=True,
            batch_num=batch_num,
        )
        return losses, prediction

    @torch.no_grad()
    def inference_step(
        self,
        images: Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        See `inference_step_with_features` for more info
        """
        prediction, _ = self.inference_step_with_features(images=images, **kwargs)
        return prediction

    def train_step_with_features(
        self,
        images: Tensor,
        targets: dict,
        predict: bool,
        batch_num: int,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict], List[torch.Tensor]]:
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

                ``"target_seg"`` (Tensor)
                    segmentation ground truth (only needed if
                    ::param::`segmenter` was provided in init) (classes start
                    from 1, 0 background)

            predict: compute final predictions (includes detection
                postprocessing)
            batch_num: batch index inside epoch

        Returns:
            Dict: all losses
            Dict: predictions for metric calculation

                ``"pred_boxes"`` List[Tensor]
                    predicted bounding boxes for each image List[[R, dim * 2]]

                ``"pred_scores"`` List[Tensor]
                    predicted probability for the class List[[R]]

                ``"pred_labels"`` List[Tensor]
                    predicted class List[[R]]

                ``"pred_seg"`` Tensor
                    predicted segmentation [N, dims]

            List[torch.Tensor]: feature maps from decoder
        """
        target_seg: Tensor = targets.get("target_seg", None)

        pred_detection, pred_seg, features = self(images)

        pred_losses = self.head.compute_loss(
            pred_detection=pred_detection,
            target_boxes=targets["target_boxes"],
            target_labels=targets["target_classes"],
            img_shape=tuple(images.shape[2:]),
        )
        if self.segmenter is not None:
            if target_seg is None:
                raise RuntimeError("Segmenter was provided to network, " "expected ground truth segmentations in step.")
            pred_losses.update(self.segmenter.compute_loss(pred_seg, target_seg))

        if predict:
            # postprocessing
            prediction = self.head.postprocess_for_inference(pred_detection, img_shape=tuple(images.shape[2:]))
            if self.segmenter is not None:
                prediction["pred_seg"] = self.segmenter.postprocess_for_inference(pred_seg)["pred_seg"]
        else:
            prediction = None
        return pred_losses, prediction, features

    def inference_step_with_features(
        self,
        images: Tensor,
        **kwargs,
    ) -> Tuple[Dict[str, Any], List[torch.Tensor]]:
        """
        Perform inference for a batch of images

        Args:
            images: batch of input images [N, C, dims]

        Returns:
            Dict: predictions

                ``"pred_boxes"`` List[Tensor]
                    predicted bounding boxes for each image List[[R, dim * 2]]

                ``"pred_scores"`` List[Tensor]
                    predicted probability for the class List[[R]]

                ``"pred_labels"`` List[Tensor]
                    predicted class List[[R]]

                ``"pred_seg"`` Tensor
                    predicted segmentation [N, C, dims]

            List[torch.Tensor]: feature maps from encoder
        """
        pred_detection, pred_seg, features = self(images)
        prediction = self.head.postprocess_for_inference(pred_detection, img_shape=tuple(images.shape[2:]))
        if self.segmenter is not None:
            prediction["pred_seg"] = self.segmenter.postprocess_for_inference(pred_seg)["pred_seg"]
        return prediction, features

    def forward(
        self,
        inp: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict, List[torch.Tensor]]:
        """
        Compute predicted bounding boxes, scores and segmentations

        Args:
            inp: batch of input images

        Returns:
            dict: predictions from head. Typically includes

                ``"pred_logits"´´ Tensor
                    predicted logits

                ``"pred_boxes"´´ Tensor
                    predicted bounding boxes in normalized center format
            Dict: semantic segmentation prediction, None if no segmenter was given
            List[torch.Tensor]: feature maps from decoder
        """
        # Compute feature list from backbone
        features = self.backbone(inp)  # [num_features] (N, C_i, px, py, (pz))
        # Reduce channel dimension with 1x1 convolution to hidden_dim
        mapped_features = self.channel_mapper(features)  # [num_feature_levels] (N, C, px, py, (pz))
        # Get Position Embedding
        pos_embeds = [self.pos_embed(feature) for feature in mapped_features]
        # transformer
        query_embed = None
        if not self.two_stage:
            query_embed = self.query_pos.weight

        out_sequence, refs_ccddcd_norm, encoder_predictions = self.transformer(
            features=mapped_features,
            query_embed=query_embed,
            pos_embed=pos_embeds,
        )
        # out_sequence: (decoder_layers or 1, bs, num_detections, hidden_dim)
        # refs_ccddcd_norm: None for DETR
        # refs_ccddcd_norm: (bs, num_detections, 3 or 6) for conditional detr
        # refs_ccddcd_norm: (decoder_layers + 1, bs, num_detections, 3 or 6) for deformable detr
        # encoder_predictions: tuple of classification and regression output of encoder

        # Calculate Boxes and Class predictions
        pred_detections = self.head(
            out_sequence=out_sequence,
            refs_ccddcd_norm=refs_ccddcd_norm,
        )

        # if a two-stage model is used, add encoder predictions to output
        if self.two_stage:
            assert encoder_predictions is not None, "Two stage is not supported by this transformer"
            pred_detections["enc_outputs"] = {
                "pred_cls_logits": encoder_predictions[0],
                "pred_box_coords": encoder_predictions[1],
            }

        # optionally forward seg head
        pred_seg = self.segmenter(features) if self.segmenter is not None else None
        return pred_detections, pred_seg, features
