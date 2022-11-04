import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from nndet.core.abstract_detr import AbstractDETR
from nndet.nn.heads.detr import BaseDETRHead
from nndet.utils.position_encoding import PositionEmbeddingSine


class BaseDETR(AbstractDETR):
    """
    Basic DETR Module, Implements forward pass, loss computation
    """

    def __init__(
        self,
        backbone,
        transformer: nn.Module,
        head: BaseDETRHead,
        hidden_dim: int,
        detection_per_img: int,
        query_dim: int,
        num_feature_levels: int = 1,
        log_queries: bool = False,
        log_ious: bool = False,
        log_features: bool = False,
    ):
        """
        Base DETR Implementation
        Args:
            backbone: Backbone network to compute image features
            transformer: Transformer Model
            head: Head used for classification, regression, loss computation and postprocessing
            hidden_dim: Dimension of the transformer sequence
            detection_per_img: number of detections the model does per patch
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
        self.total_feature_levels = len(channels)

        # For future multi feature
        if num_feature_levels == 1:
            self.input_proj = nn.ModuleList(
                [nn.Conv3d(channels[-1], self.hidden_dim, kernel_size=1)]
            )
        else:
            raise NotImplementedError

        # Build Transformer Specific Architecture
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=self.hidden_dim)
        self.transformer = transformer
        self.decoder_layers = transformer.dec_layers
        self.query_pos = nn.Embedding(detection_per_img, query_dim)

        # Build the final layers for classification and box regression
        self.head = head

        # toggle the debug mode
        self.log_query = log_queries
        self.log_iou = log_ious
        self.log_features = log_features

    def forward(
        self,
        inp: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], List, Dict, List[torch.Tensor]]:
        """
        Compute predicted bounding boxes, scores and segmentations

        Args:
            inp (torch.Tensor): batch of input images

        Returns:
            dict: predictions from head. Typically includes

                ``"pred_logits"´´ Tensor of predicted logits
                ``"pred_boxes"´´ Tensor of predicted bounding boxes in normalized center format

            List[torch.Tensor]: list of anchors, empty list for DETR

            dict: segmentation prediction. None, for segmentation use DETRSegmentation

            List[torch.Tensor]: feature maps from decoder
        """

        # Compute feature list from backbone
        features = self.backbone(inp)  # [l] (N, C_i, px, py, pz)
        # Reduce channel dimension with 1x1 convolution to hidden_dim
        srcs_sequence = self.input_proj[0](features[-1]).unsqueeze(
            dim=1
        )  # (N, 1, C, px, py, pz)
        # Get Position Embedding and pass through transformer
        pos_embed = self.pos_embed(srcs_sequence.squeeze(dim=1))  # (N, C, px, py, pz)
        out_sequence, memory, reference = self.transformer(
            srcs_sequence, self.query_pos.weight, pos_embed
        )
        # out_sequence: (decoder_layers or 1, bs, num_detections, hidden_dim)
        # memory: (bs, hidden_dim, h/stride, w/stride, d/stride): used for segmentation head
        # reference: (bs, num_detections, 3 or 6) or None: used for bounding box calculation

        # Calculate Boxes and Class predictions
        pred_detections = self.head(out_sequence, reference)
        return pred_detections, [], {}, features

    def forward_log_features_and_attention_maps(
        self, images: Tensor, targets: Dict, batch_num: int
    ) -> Tuple[Dict[str, torch.Tensor], List, Dict, List[torch.Tensor]]:
        """
        Sets forward hooks to save features, queries and attention maps, then do forward step and save
        See `self.forward` for more info
        """
        # Set some hooks for visualization
        (
            conv_features,
            enc_attn_weights,
            dec_attn_weights,
            enc_features,
            dec_queries0,
            dec_queries1,
        ) = ([], [], [], [], [], [])
        hooks = [
            self.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                lambda _self, _input, output: enc_attn_weights.append(output[1])
            ),
            self.transformer.decoder.layers[-1].cross_attn.register_forward_hook(
                lambda _self, _input, output: dec_attn_weights.append(output[1])
            ),
            self.transformer.encoder.register_forward_hook(
                lambda _self, _input, output: enc_features.append(output)
            ),
            self.input_proj[0].register_forward_hook(
                lambda _self, _input, output: conv_features.append(output)
            ),
            self.transformer.decoder.layers[0].register_forward_hook(
                lambda _self, _input, output: dec_queries0.append(output)
            ),
            self.transformer.decoder.layers[1].register_forward_hook(
                lambda _self, _input, output: dec_queries1.append(output)
            ),
        ]
        # Forward pass and loss computation
        pred_detection, _, pred_seg, features = self(images)

        for hook in hooks:
            hook.remove()
        conv_features = conv_features[0]
        enc_attn_weights = enc_attn_weights[0]
        dec_attn_weights = dec_attn_weights[0]
        enc_features = enc_features[0]
        dec_queries0 = dec_queries0[0]
        dec_queries1 = dec_queries1[0]
        if not os.path.exists("vis"):
            os.mkdir("vis")
        torch.save(images, f"vis/images{batch_num}.pt")
        torch.save(targets, f"vis/conv_features{batch_num}.pt")
        torch.save(conv_features, f"vis/conv_features{batch_num}.pt")
        torch.save(enc_attn_weights, f"vis/enc_attn{batch_num}.pt")
        torch.save(dec_attn_weights, f"vis/dec_attn{batch_num}.pt")
        torch.save(enc_features, f"vis/enc_features{batch_num}.pt")
        torch.save(dec_queries0, f"vis/dec_queries0{batch_num}.pt")
        torch.save(dec_queries1, f"vis/dec_queries1{batch_num}.pt")
        return pred_detection, _, pred_seg, features

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
        if not self.log_features:
            pred_detection, _, pred_seg, features = self(images)
        else:
            (
                pred_detection,
                _,
                pred_seg,
                features,
            ) = self.forward_log_features_and_attention_maps(images, targets, batch_num)

        # Log the predicted queries
        if self.log_query:
            self.log_queries(
                targets["target_classes"], pred_detection["pred_logits"], batch_num
            )

        pred_losses, _ = self.head.compute_loss(
            pred_detection, targets, images.shape[2:]
        )

        if predict:
            # postprocessing
            prediction = self.postprocess_for_inference(images, pred_detection)
            if self.log_iou:
                self.log_ious(pred_detection["pred_boxes"])
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
        pred_detection, anchors, pred_seg, features = self(images)
        prediction = self.postprocess_for_inference(
            images=images,
            pred_detection=pred_detection,
            anchors=anchors,
            pred_seg=pred_seg,
        )
        return prediction, features
