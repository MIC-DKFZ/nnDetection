from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from nndet.core.abstract import AbstractDetector
from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.heads.detr import BaseDETRHead
from nndet.nn.heads.segmenter import Segmenter
from nndet.nn.layers.pos_embed.sine import BasePositionEmbedding


class BaseDETR(AbstractDetector):
    def __init__(
        self,
        backbone: AbstractBackbone,
        transformer: nn.Module,
        head: BaseDETRHead,
        pos_embed: BasePositionEmbedding,
        hidden_dim: int,
        detection_per_img: int,
        query_dim: int,
        num_feature_levels: int = 1,
        segmenter: Optional[Segmenter] = None,
        # debugging
        log_queries: bool = False,
        log_ious: bool = False,
        log_features: bool = False,
    ):
        """
        Basic DETR Module, Implements forward pass, loss computation

        Args:
            backbone: Backbone network to compute image features
            transformer: Transformer Model
            head: Head used for classification, regression, loss computation and postprocessing
            hidden_dim: Dimension of the transformer sequence
            detection_per_img: number of detections the model does per patch
            pos_embed: module to generate positional embedding
            query_dim: dimension of object queries in the decoder (usually same as hidden dim except for DABDETR)
            num_feature_levels: which levels of backbone input should be used for the transformer input
                                (currently only one supported)
            log_queries: log the predicted queries for analysis
            log_ious: log the ious of predicted boxes for analysis
            log_features: log features or attention maps
        """
        super().__init__()
        # Obtain important hyperparameters
        self.detection_per_img = detection_per_img
        # Set Backbone and get channels and feature levels
        self.backbone = backbone
        channels = self.backbone.get_channels()
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels

        # For future multi feature
        if num_feature_levels == 1:
            self.input_proj = nn.ModuleList([nn.Conv3d(channels[-1], self.hidden_dim, kernel_size=1)])
        else:
            raise NotImplementedError

        # Build Transformer Specific Architecture
        self.pos_embed = pos_embed
        self.transformer = transformer
        self.decoder_layers = transformer.dec_layers
        self.query_pos = nn.Embedding(detection_per_img, query_dim)

        # Build the final layers for classification and box regression
        self.head = head

        # Build optional modules
        self.segmenter = segmenter

        # toggle the debug mode
        self.log_query = log_queries
        self.log_iou = log_ious
        self.log_features = log_features

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
                    segmentation ground truth (only needed if ::param::`segmenter`
                    was provided in init) (classes start from 1, 0 background)

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
        pred_losses, _ = self.head.compute_loss(pred_detection, targets, images.shape[2:])

        if self.segmenter is not None:
            if target_seg is None:
                raise RuntimeError("Segmenter was provided to network, " "expected ground truth segmentations in step.")
            pred_losses.update(self.segmenter.compute_loss(pred_seg, target_seg))

        if predict:
            # postprocessing
            prediction = self.head.postprocess_for_inference(images, pred_detection)
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
            images: batch of input images [N, C, W, H, (D)]

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
        prediction = self.head.postprocess_for_inference(images, pred_detection)
        if self.segmenter is not None:
            prediction["pred_seg"] = self.segmenter.postprocess_for_inference(pred_seg)["pred_seg"]
        return prediction, features

    def forward(
        self,
        inp: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], List, Dict, List[torch.Tensor]]:
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

            List[torch.Tensor]: list of anchors, empty list for DETR
            Dict: segmentation prediction. None, for segmentation use DETRSegmentation
            List[torch.Tensor]: feature maps from decoder
        """

        # Compute feature list from backbone
        features = self.backbone(inp)  # [l] (N, C_i, px, py, pz)
        # Reduce channel dimension with 1x1 convolution to hidden_dim
        srcs_sequence = self.input_proj[0](features[-1]).unsqueeze(dim=1)  # (N, 1, C, px, py, pz)
        # Get Position Embedding and pass through transformer
        pos_embed = self.pos_embed(srcs_sequence.squeeze(dim=1))  # (N, C, px, py, pz)
        out_sequence, memory, reference = self.transformer(srcs_sequence, self.query_pos.weight, pos_embed)
        # out_sequence: (decoder_layers or 1, bs, num_detections, hidden_dim)
        # memory: (bs, hidden_dim, h/stride, w/stride, d/stride): used for segmentation head
        # reference: (bs, num_detections, 3 or 6) or None: used for bounding box calculation

        # Calculate Boxes and Class predictions
        pred_detections = self.head(out_sequence, reference)

        # optionally forward seg head
        pred_seg = self.segmenter(features) if self.segmenter is not None else None
        return pred_detections, pred_seg, features
