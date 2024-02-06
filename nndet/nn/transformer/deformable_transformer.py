# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from detrex licensed under
# SPDX-FileCopyrightText: 2022, The IDEA Authors
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from nndet.nn.heads.classifier.ffn import FFNClassifier
from nndet.nn.heads.regressor.ffn import FFNRegressor
from nndet.nn.transformer.attention.multi_scale_deform_attn_3d import (
    MultiScaleDeformableAttention,
)
from nndet.nn.transformer.layers.abstract import (
    BaseTransformerDecoder,
    BaseTransformerEncoder,
)


class DeformableDETRTransformer(nn.Module):
    def __init__(
        self,
        encoder: BaseTransformerEncoder,
        decoder: BaseTransformerDecoder,
        classifier: Optional[FFNClassifier] = None,
        regressor: Optional[FFNRegressor] = None,
        num_feature_levels: int = 4,
        two_stage: bool = False,
        two_stage_num_proposals: int = 300,  # TODO: add to config
        two_stage_base_object_scale: float = 0.05,  # TODO: add to config
        pos_embed_temperature: float = 10000,  # TODO: add to config
    ):
        """
        Transformer module for Deformable DETR

        Args:
            encoder: encoder module.
            decoder: decoder module.
            classifier: mlp to calculate the class of the encoder
                predictions (needs to be sigmoid based)
            regressor:  mlp to calculate the boxes of the encoder
                predictions
            two_stage: whether to use encoder predictions as initialization
                for the decoder
            num_feature_levels: number of feature levels
            two_stage_num_proposals: number of proposals in two-stage
                transformer
            two_stage_base_object_scale: base object scale for two-stage
                deformable DETR (Formula can be found in Deformable DETR
                paper, appendix 'Two-Stage Deformable DETR').
            pos_embed_temperature: temperature for positional embedding

        Warning:
            Only sigmoid based classifiers are supported right now!
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dim = self.encoder.embed_dim

        self.num_feature_levels = num_feature_levels
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.two_stage_base_object_scale = two_stage_base_object_scale
        self.pos_embed_num_feats = self.embed_dim // 2
        self.pos_embed_temperature = pos_embed_temperature

        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dim))

        if self.two_stage:
            self.enc_output = nn.Linear(self.embed_dim, self.embed_dim)
            self.enc_output_norm = nn.LayerNorm(self.embed_dim)
            self.pos_trans = nn.Linear(self.embed_dim * 3, self.embed_dim * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dim * 2)
        else:
            self.reference_points = nn.Linear(self.embed_dim, 3)

        self.init_weights()
        # Classifier and regressor are already initialized
        self.classifier = classifier
        self.regressor = regressor

    def init_weights(self):
        """
        Initialize the weights of the transformer modules
        """
        for n, p in self.named_parameters():
            if p.dim() > 1 and "regressor" not in n:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if not self.two_stage:
            nn.init.xavier_normal_(self.reference_points.weight.data, gain=1.0)
            nn.init.constant_(self.reference_points.bias.data, 0.0)
        nn.init.normal_(self.level_embeds)

    def forward(
        self,
        features: List[torch.Tensor],
        query_embed: torch.Tensor,
        pos_embed: List[torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Compute the output box embeddings given the input features, position
        embedding and query embedding

        Args:
            features: features from the backbone in form of a
                List[Tensor(bs, C, dims)]
            query_embed: object queries = input for the transformer decoder
                [num_queries, C], where C is the embedding dimension * 2
                (will be splitted for normal and pos queries!) and
                num_queries is the number of object queries (predictions).
                Only used if `two_stage` is `False`.
            pos_embed: position embedding for the features, same shape as
                features

        Returns:
            Tensor: output box embeddings (output of the decoder)
                ((num_decoder_layers), bs, num_queries, C) where
                `num_decoder_layers` is the number of decoder layers, bs is
                the batch size, num_queries is the number of object queries,
                and C is the number of channels in the transformer
            Tensor: references from the transformer decoder. If `two_stage`
                will be  of shape (num_decoder_layers + 1, bs, num_queries,
                dim * 2) where `num_decoder_layers` is the number of decoder
                layers, bs is the batch size, num_queries is the number of
                object queries, and C is the number of channels in the
                transformer. If not `two_stage` is `False`, it will contain
                points of shape (num_decoder_layers + 1, bs, num_queries,
                dim). Reference points are normed to [0, 1] and in
                center format (cx, cy, cz, dx, dy, dz).
            Optional(Tensor): if `two_stage` is `False` returns None.
                if `two_stage` is `True` returns the output of the
                encoder head which is a tuple where the first entry
                corresponds to the classification output (shape
                [bs, s-dims, num_classes]) and the second is the regressor
                output (shape [bs, s-dims, dims * 2]), where bs
                is the batch size, s-dims is the sum of the number of
                pixels across the multi-scale feature maps, and dims
                is the number of spatial dimensions. The coordinates
                are inverted with respect to the regressor non-linearity!
        """
        assert len(features) == len(pos_embed)
        assert self.two_stage or query_embed is not None

        feat_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        for lvl, (feat, pos_embed_feat) in enumerate(zip(features, pos_embed)):
            bs, _, ax0, ax1, ax2 = feat.shape
            spatial_shapes.append((ax0, ax1, ax2))  # permute
            feat = feat.flatten(2).transpose(1, 2)  # bs, embed_dim, p-dims -> bs, p-dims, embed_dim
            feat_flatten.append(feat)

            # bs, embed_dim, p-dims -> bs, p-dims, embed_dim
            pos_embed_feat = pos_embed_feat.flatten(2).transpose(1, 2)
            # num_level, embed_dim -> 1, 1, embed_dim -> 1, p-dims, embed_dim
            lvl_pos_embed = pos_embed_feat + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)

        feat_flatten = torch.cat(feat_flatten, 1)  # bs, level * p-dims, embed_dim
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # 1, level * p-dims, embed_dim
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)  # nlvl, 3
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))  # nlvl

        # (bs, p-dims, num_levels, 3) , normalized coordinated
        reference_points = self.get_reference_points(spatial_shapes, batch_size=bs, device=feat_flatten[-1].device)

        memory = self.encoder(
            query=feat_flatten,  # bs, level * p-dims, embed_dim
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,  # bs, level * p-dims, embed_dim
            key_pos=None,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,  # bs, num_token, num_level, 2
            level_start_index=level_start_index,
            attn_masks=None,
            query_key_padding_mask=None,
            key_padding_mask=None,
            **kwargs,
        )

        bs, _, c = memory.shape
        if self.two_stage:
            # output_memory: bs, num_tokens, c
            # output_proposals: bs, num_tokens, 6. coords non-lin-inverted.
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, spatial_shapes)

            # bs, num_tokens, num_classes
            enc_outputs_class = self.classifier.encoder_mlp(output_memory)
            # bs, num_tokens, dims: coords non-lin-inverted.
            enc_outputs_coord_unact = self.regressor.encoder_mlp(output_memory) + output_proposals

            # bs, num_tokens, num_classes -> bs, num_tokens -> topk(1) -> bs, topk: topk indices as tensor
            topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], self.two_stage_num_proposals, dim=1)[1]

            # bs, topk, dims
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 6))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = self.regressor.apply_non_lin(topk_coords_unact)  # normalized coords, actually boxes
            init_reference_out = reference_points

            pos_trans_out = self.get_proposal_pos_embed(topk_coords_unact)
            pos_trans_out = self.pos_trans_norm(self.pos_trans(pos_trans_out))
            query_pos, query = torch.split(pos_trans_out, c, dim=2)
        else:
            # If not using two stage: use the given query embed for content and position queries
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.regressor.apply_non_lin(self.reference_points(query_pos))
            init_reference_out = reference_points

        # decoder
        inter_states, inter_references = self.decoder(
            query=query,  # bs, num_queries, embed_dims
            key=None,  # bs, num_tokens, embed_dims
            value=memory,  # bs, num_tokens, embed_dims
            query_pos=query_pos,
            key_pos=query_pos,
            reference_points=reference_points,  # num_queries, 6
            spatial_shapes=spatial_shapes,  # nlvl, 2
            level_start_index=level_start_index,  # nlvl
            attn_masks=None,
            query_key_padding_mask=None,
            key_padding_mask=None,
            **kwargs,
        )

        # Concatenate references into one array
        reference_out = torch.cat([init_reference_out.unsqueeze(0), inter_references], dim=0)
        if self.two_stage:
            return inter_states, reference_out, (enc_outputs_class, enc_outputs_coord_unact)
        else:
            return inter_states, reference_out, None

    @staticmethod
    def get_reference_points(
        spatial_shapes: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Get the reference points used in decoder.

        Args:
            spatial_shapes: the shape of all feature maps,
                has shape (num_level, 3).
            batch_size: the batch size of the input data.
            device: the device where reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has shape
                (bs, p-dims, num_levels, 3). Points are normalized.
        """
        num_level = len(spatial_shapes)
        reference_points_list = []
        for lvl, (ax0_shape, ax1_shape, ax2_shape) in enumerate(spatial_shapes):
            ref_ax0, ref_ax1, ref_ax2 = torch.meshgrid(
                torch.linspace(0.5, ax0_shape - 0.5, ax0_shape, dtype=torch.float32, device=device),
                torch.linspace(0.5, ax1_shape - 0.5, ax1_shape, dtype=torch.float32, device=device),
                torch.linspace(0.5, ax2_shape - 0.5, ax2_shape, dtype=torch.float32, device=device),
                indexing="ij",
            )

            ref_ax0 = (ref_ax0.reshape(-1))[None].expand(batch_size, -1) / ax0_shape
            ref_ax1 = (ref_ax1.reshape(-1))[None].expand(batch_size, -1) / ax1_shape
            ref_ax2 = (ref_ax2.reshape(-1))[None].expand(batch_size, -1) / ax2_shape

            ref = torch.stack((ref_ax0, ref_ax1, ref_ax2), -1)  # bs, p-dims, 3
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)  # bs, p-dims, 3
        # bs, p-dims, num_levels, 3
        reference_points = reference_points[:, :, None].expand(-1, -1, num_level, -1)
        return reference_points

    def gen_encoder_output_proposals(
        self,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate an initial set of proposals and process the encoder output
        with ``enc_output`` and ``enc_output_norm``.

        Args:
            memory: output from transformer encoder, shape
                [bs, ref_points, embed_dim], where ref_points is the number
                of reference points in the encoder (usually the sum of the
                number of pixels across the multi-scale feature maps)
            spatial_shapes: the shape of all feature maps, has shape
                (num_level, 3).

        Returns:
            torch.Tensor: processed encoder output, shape
                [bs, s-dims, embed_dim], where  bs is the batch size, s-dims
                is the sum of the number of pixels across the multi-scale
                feature maps, and embed_dim is the embedding dimension.
            torch.Tensor: proposals for encoder prediction in center-format
                (cx, cy, cz, dx, dy, dz) with shape [bs, s-dims, 2 * dims], where
                bs is the batch size, s-dims is the sum of the number of
                pixels across the multi-scale feature maps, and dims
                is the number of spatial dimensions. The coordinates
                are inverted with respect to the regressor non-linearity!
        """
        BS, _, _ = memory.shape
        proposals = []
        for lvl, (ax0, ax1, ax2) in enumerate(spatial_shapes):
            grid_ax0, grid_ax1, grid_ax2 = torch.meshgrid(
                torch.linspace(0, ax0 - 1, ax0, dtype=torch.float32, device=memory.device),
                torch.linspace(0, ax1 - 1, ax1, dtype=torch.float32, device=memory.device),
                torch.linspace(0, ax2 - 1, ax2, dtype=torch.float32, device=memory.device),
                indexing="ij",
            )
            grid = torch.stack([grid_ax0, grid_ax1, grid_ax2], -1)  # ax0, ax1, ax2, 3
            scale = (
                torch.as_tensor([[ax0, ax1, ax2]], dtype=torch.float32, device=memory.device)
                .expand(BS, 3)
                .view(BS, 1, 1, 1, 3)
            )  # bs, 1, 1, 1, 3
            grid = (grid.unsqueeze(0).expand(BS, -1, -1, -1, -1) + 0.5) / scale
            ax012 = torch.ones_like(grid) * self.two_stage_base_object_scale * (2.0**lvl)
            proposal = torch.cat((grid, ax012), -1).view(BS, -1, 6)
            proposals.append(proposal)

        # proposals
        output_proposals = torch.cat(proposals, 1)  # bs, s-dims, 6
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = self.regressor.apply_inverse_non_lin(output_proposals, eps=0)
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        # memory
        output_memory = memory
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_proposal_pos_embed(
        self,
        proposals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the position embedding of the proposal.

        Args:
            proposals: proposals for encoder prediction in center-format
                (cx, cy, cz, dx, dy, dz) with shape [bs, R, 2 * dims], where
                bs is the batch size, R is the number of proposals, and dims
                is the number of spatial dimensions. Proposals should be
                non-lin-inverted (not normalised) coordinates.

        Returns:
            torch.Tensor: positional embedding of proposals, shape
                [bs, R, `self.pos_embed_num_feats` * 2 * dims], where bs is the batch size, R
                is the number of proposals
        """
        dim_t = torch.arange(self.pos_embed_num_feats, dtype=torch.float32, device=proposals.device)
        dim_t = self.pos_embed_temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.pos_embed_num_feats
        )

        # bs, R, 2 * dims: normalized coords
        proposals = self.regressor.apply_non_lin(proposals) * 2 * np.pi
        pos = proposals[..., None] / dim_t  # bs, R, 2 * dims, num_pos_feats
        pos = torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), dim=-1).flatten(2)
        return pos
