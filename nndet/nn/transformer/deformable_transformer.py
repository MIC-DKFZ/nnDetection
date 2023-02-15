# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from nndet.nn.transformer.attention.multi_scale_deform_attn_3d import (
    MultiScaleDeformableAttention,
)
from nndet.nn.transformer.layers.base_layer import TransformerLayerSequence


class DeformableDETRTransformer(nn.Module):
    def __init__(
        self,
        encoder: TransformerLayerSequence,
        decoder: TransformerLayerSequence,
        num_feature_levels: int = 4,
        as_two_stage: bool = False,
        two_stage_num_proposals: int = 300,
    ):
        """
        Transformer module for Deformable DETR

        Args:
            encoder: encoder module.
            decoder: decoder module.
            as_two_stage: whether to use two-stage transformer
            num_feature_levels: number of feature levels
            two_stage_num_proposals: number of proposals in two-stage
                transformer
        """
        super(DeformableDETRTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_feature_levels = num_feature_levels
        self.as_two_stage = as_two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        self.embed_dim = self.encoder.embed_dim

        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dim))

        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dim, self.embed_dim)
            self.enc_output_norm = nn.LayerNorm(self.embed_dim)
            self.pos_trans = nn.Linear(self.embed_dim * 3, self.embed_dim * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dim * 2)
        else:
            self.reference_points = nn.Linear(self.embed_dim, 3)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if not self.as_two_stage:
            nn.init.xavier_normal_(self.reference_points.weight.data, gain=1.0)
            nn.init.constant_(self.reference_points.bias.data, 0.0)
        nn.init.normal_(self.level_embeds)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N, S, C = memory.shape
        proposals = []
        _cur = 0
        for lvl, (D, H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + D * H * W)].view(N, D, H, W, 1)
            valid_D = torch.sum(~mask_flatten_[:, :, 0, 0, 0], 1)
            valid_H = torch.sum(~mask_flatten_[:, 0, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, 0, :, 0], 1)
            grid_x, grid_y, grid_z = torch.meshgrid(
                torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device),
                torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(0, D - 1, D, dtype=torch.float32, device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1), grid_z.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1), valid_D.unsqueeze(-1)], 1).view(
                N, 1, 1, 1, 3
            )
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1, -1) + 0.5) / scale
            whd = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, whd), -1).view(N, -1, 6)
            proposals.append(proposal)
            _cur += W * H * D

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 3).
            valid_ratios (Tensor): The ratios of valid
                points on the feature map, has shape
                (bs, num_levels, 3)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 3).
        """
        reference_points_list = []
        for lvl, (D, H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_x, ref_y, ref_z = torch.meshgrid(
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, D - 0.5, D, dtype=torch.float32, device=device),
            )
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_z = ref_z.reshape(-1)[None] / (valid_ratios[:, None, lvl, 2] * D)

            ref = torch.stack((ref_x, ref_y, ref_z), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    @staticmethod
    def get_valid_ratio(mask):
        _, D, H, W = mask.shape
        valid_D = torch.sum(~mask[:, :, 0, 0], 1)
        valid_H = torch.sum(~mask[:, 0, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, 0, :], 1)

        valid_ratio_d = valid_D.float() / D
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h, valid_ratio_d], -1)
        return valid_ratio

    def get_proposal_pos_embed(self, proposals, num_pos_feats=64, temperature=10000):
        """
        Get the position embedding of proposal.
        """
        scale = 2 * np.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
        # N, L, 6
        proposals = proposals.sigmoid() * scale
        # N, L, 6, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 6, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def forward(
        self,
        features: List[torch.Tensor],
        query_embed: torch.Tensor,
        pos_embed: List[torch.Tensor],
        mask: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Compute the output box embeddings given the input features, position
        embedding and query embedding

        Args:
            features: features from the backbone in form of a
            List[Tensor(bs, C, H, W, (Z))]
            query_embed: object queries = input for the transformer decoder
            pos_embed: position embedding for the features, same shape as
                features
            mask: mask to mask out certain pixels of the feature maps, same
                shape as features

        Returns:
            Tensor: output box embeddings (output of the decoder)
                ((num_decoder_layers), bs, num_queries, C)
            Tensor: references from the transformer decoder
            Optional(Tensor): References from the transformer decoder
                ((num_decoder_layers), bs, num_queries, dim)
        """
        assert self.as_two_stage or query_embed is not None
        feat_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        mask_list = [] if mask is None else mask
        mask_flatten = []
        # Permute to d, h, w format
        for lvl, (feat, pos_embed) in enumerate(zip(features, pos_embed)):
            bs, c, w, h, d = feat.shape
            spatial_shape = (d, h, w)
            spatial_shapes.append(spatial_shape)
            if mask is None:
                mask_list.append(torch.zeros((bs, d, h, w)))
                mask_flatten.append(mask_list[lvl].flatten(2))  # bs, dhw
            else:
                mask_flatten.append(mask_list[lvl].permute(0, 3, 2, 1).flatten(1))  # bs, dhw
            feat = feat.permute(0, 1, 4, 3, 2).flatten(2).transpose(1, 2)  # bs, dhw, c
            feat_flatten.append(feat)
            pos_embed = pos_embed.permute(0, 1, 4, 3, 2).flatten(2).transpose(1, 2)  # bs, dhw, c
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)

        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in mask_list], 1)
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=feat_flatten[-1].device)

        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,  # bs, num_token, num_level, 2
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )
        bs, _, c = memory.shape
        if self.as_two_stage:

            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            # output_memory: bs, num_tokens, c
            # output_proposals: bs, num_tokens, 6. unsigmoided.

            enc_outputs_class = self.decoder.class_embed(
                output_memory,
                layer=self.decoder.num_layers,
            )
            enc_outputs_coord_unact = (
                self.decoder.bbox_embed(output_memory, self.decoder.num_layers) + output_proposals
            )  # unsigmoided.

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]

            # extract region proposal boxes
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 6)
            )  # unsigmoided.
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact, num_pos_feats=self.embed_dim // 2))
            )
            query_pos, query = torch.split(pos_trans_out, c, dim=2)
        else:
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos).sigmoid()
            init_reference_out = reference_points

        # decoder
        inter_states, inter_references = self.decoder(
            query=query,  # bs, num_queries, embed_dims
            key=None,  # bs, num_tokens, embed_dims
            value=memory,  # bs, num_tokens, embed_dims
            query_pos=query_pos,
            key_padding_mask=mask_flatten,  # bs, num_tokens
            reference_points=reference_points,  # num_queries, 6
            spatial_shapes=spatial_shapes,  # nlvl, 2
            level_start_index=level_start_index,  # nlvl
            valid_ratios=valid_ratios,  # bs, nlvl, 2
            **kwargs,
        )
        reference_out = torch.cat([init_reference_out.unsqueeze(0), inter_references], dim=0)
        if self.as_two_stage:
            return inter_states, reference_out, (enc_outputs_class, enc_outputs_coord_unact)
        else:
            return inter_states, memory, reference_out, None
