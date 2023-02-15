from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from nndet.core.detr import BaseDETR
from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.heads.detr.base import DETRHead
from nndet.nn.heads.segmenter import Segmenter
from nndet.nn.layers.pos_embed.sine import BasePositionEmbedding


class DeformableDETR(BaseDETR):
    def __init__(
        self,
        backbone: AbstractBackbone,
        transformer: nn.Module,
        head: DETRHead,
        pos_embed: BasePositionEmbedding,
        hidden_dim: int,
        detection_per_img: int,
        query_dim: int,
        num_feature_levels: int = 4,
        segmenter: Optional[Segmenter] = None,
        box_refine: bool = True,
        two_stage: bool = True,
    ):
        # Deformable DETR uses query dim for position and content embedding so 2 times the size
        query_dim *= 2
        super().__init__(
            backbone,
            transformer,
            head,
            pos_embed,
            hidden_dim,
            detection_per_img,
            query_dim,
            num_feature_levels,
            segmenter,
        )
        self.box_refine = box_refine
        self.two_stage = two_stage
        # two-stage
        self.transformer.decoder.bbox_embed = self.head.regressor if two_stage else None
        self.transformer.decoder.class_embed = self.head.classifier if two_stage else None

    def forward(
        self,
        inp: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], List, Dict, List[torch.Tensor]]:
        """
        Compute predicted bounding boxes, scores and segmentations

        Args:
            inp (torch.Tensor): batch of input images
            targets (Dict): ground truth dict

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
        if self.num_feature_levels == 1:
            # Reduce channel dimension with 1x1 convolution to hidden_dim
            multi_level_features = self.input_proj[0](features[-1]).unsqueeze(dim=1)  # (N, 1, C, px, py, pz)
        else:
            multi_level_features = []
            for i in range(self.num_feature_levels):
                channel_idx = self.input_feature_levels - self.num_feature_levels + i
                # Deformable Attention needs D, H, W format
                multi_level_features.append(
                    self.input_proj[i](features[channel_idx]).permute(0, 1, 4, 3, 2)
                )  # (N, l, C, ?, ?, ?)

        multi_level_masks = []
        multi_level_position_embeddings = []
        for feature in multi_level_features:
            # Get Position Embedding, permute for D H W format
            multi_level_position_embeddings.append(
                self.pos_embed(feature.permute(0, 1, 4, 3, 2)).permute(0, 1, 4, 3, 2)
            )  # (N, C, pz, py, px)
            multi_level_masks.append(
                torch.zeros(
                    feature.size(0),
                    feature.size(2),
                    feature.size(3),
                    feature.size(4),
                    dtype=torch.bool,
                    device=feature.device,
                )
            )
        # initialize object query embeddings
        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_pos.weight
        # feed into transformer
        (
            inter_states,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = self.transformer(
            multi_level_features,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embeds,
        )

        # Calculate Boxes and Class predictions
        pred_detections = self.head(inter_states, inter_references)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            pred_detections["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            }

        # optionally forward seg head
        pred_seg = self.segmenter(features) if self.segmenter is not None else None
        return pred_detections, pred_seg, features
