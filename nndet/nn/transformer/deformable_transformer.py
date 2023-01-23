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
import numpy as np
import torch
import torch.nn as nn
from nndet_detr.util.misc import FFN, inverse_sigmoid

from nndet.nn.transformer.attention.multi_head_attention_wrapper import (
    MultiheadAttention,
)
from nndet.nn.transformer.attention.multi_scale_deform_attn_3d import (
    MultiScaleDeformableAttention,
)
from nndet.nn.transformer.base_layer.base_transformer_layer import BaseTransformerLayer
from nndet.nn.transformer.base_layer.transformer_layer_sequence import (
    TransformerLayerSequence,
)


class DeformableDETRTransformerEncoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        num_layers: int = 6,
        post_norm: bool = False,
        num_feature_levels: int = 4,
        num_points: int = 4,
    ):
        super(DeformableDETRTransformerEncoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=[
                    MultiScaleDeformableAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        dropout=attn_dropout,
                        batch_first=True,
                        num_levels=num_feature_levels,
                        num_points=num_points,
                    )
                ],
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    num_fcs=2,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),
                operation_order=("self_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):

        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class DeformableDETRTransformerDecoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        num_layers: int = 6,
        return_intermediate: bool = True,
        num_feature_levels: int = 4,
        num_points: int = 4,
    ):
        super(DeformableDETRTransformerDecoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=[
                    MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=True,
                    ),
                    MultiScaleDeformableAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        dropout=attn_dropout,
                        batch_first=True,
                        num_levels=num_feature_levels,
                        num_points=num_points,
                    ),
                ],
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),
                operation_order=(
                    "self_attn",
                    "norm",
                    "cross_attn",
                    "norm",
                    "ffn",
                    "norm",
                ),
            ),
            num_layers=num_layers,
        )
        self.return_intermediate = return_intermediate

        self.bbox_embed = None
        self.class_embed = None

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        reference_points=None,  # num_queries, 4. normalized.
        valid_ratios=None,
        **kwargs,
    ):
        output = query

        intermediate = []
        intermediate_reference_points = []
        for layer_idx, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 6:
                reference_points_input = (
                    reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 3
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points_input,
                **kwargs,
            )

            if self.bbox_embed is not None:
                tmp = self.bbox_embed(output, layer_idx)
                # FIXME the order xyz,whd might be wrong here
                if reference_points.shape[-1] == 6:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 3
                    new_reference_points = tmp
                    new_reference_points[..., :3] = tmp[..., :3] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


class DeformableDETRTransformer(nn.Module):
    """Transformer module for DINO
    Args:
        encoder (nn.Module): encoder module.
        decoder (nn.Module): decoder module.
        as_two_stage (bool): whether to use two-stage transformer. Default False.
        num_feature_levels (int): number of feature levels. Default 4.
        two_stage_num_proposals (int): number of proposals in two-stage transformer. Default 900.
    """

    def __init__(
        self,
        encoder=None,
        decoder=None,
        num_feature_levels=4,
        as_two_stage=False,
        two_stage_num_proposals=300,
    ):
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
        """Get the reference points used in decoder.
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
        """Get the position embedding of proposal."""
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
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        query_embed,
        **kwargs,
    ):
        assert self.as_two_stage or query_embed is not None
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)
        ):
            bs, c, d, h, w = feat.shape
            spatial_shape = (d, h, w)
            spatial_shapes.append(spatial_shape)

            feat = feat.flatten(2).transpose(1, 2)  # bs, dhw, c
            mask = mask.flatten(1)  # bs, dhw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, dhw, c
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)

        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)

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

        inter_references_out = inter_references
        if self.as_two_stage:
            return (
                inter_states,
                init_reference_out,
                inter_references_out,
                enc_outputs_class,
                enc_outputs_coord_unact,
            )
        else:
            return inter_states, init_reference_out, inter_references_out, None, None
