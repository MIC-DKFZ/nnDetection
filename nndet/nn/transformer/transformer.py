# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from detrex licensed under
# SPDX-FileCopyrightText: 2022, The IDEA Authors
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from nndet.nn.transformer.abstract_transformer import AbstractTransformer
from nndet.nn.transformer.layers.base_layer import TransformerLayerSequence


class DETRTransformer(AbstractTransformer):
    def __init__(
        self,
        encoder: TransformerLayerSequence,
        decoder: TransformerLayerSequence,
    ):
        """
        Transformer module for DETR.
        Args:
            encoder: Transformer encoder
            decoder: Transformer decoder
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dim = self.encoder.embed_dim
        self.dim = decoder.dim
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        features: List[torch.Tensor],
        query_embed: torch.Tensor,
        pos_embed: List[torch.Tensor],
        mask: Optional[List[torch.Tensor]] = None,
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

        # This is single resolution so unpack the multiscale lists
        assert len(features) == len(pos_embed) == 1
        features, pos_embed = features[0], pos_embed[0]
        dim = self.decoder.dim
        assert dim == features.dim() - 2  # subtract batch size and sequence length
        if dim == 2:
            bs, c, h, w = features.shape
        elif dim == 3:
            bs, c, h, w, z = features.shape
        else:
            raise ValueError(f"Number of feature dimensions {dim} not supported.")

        features = features.view(bs, c, -1).permute(2, 0, 1)  # [bs, c, h, w] -> [h*w*z, bs, c]
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]

        if mask is not None:
            assert len(mask) == 0
            mask = mask[0].view(bs, -1)  # [bs, h, w] -> [bs, h*w]

        memory = self.encoder(
            query=features,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask,
        )

        target = torch.zeros_like(query_embed)
        hidden_state, references = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
        )
        hidden_state = hidden_state.transpose(1, 2)
        if dim == 4:
            memory = memory.permute(1, 2, 0).reshape(bs, c, h, w)
        else:
            memory = memory.permute(1, 2, 0).reshape(bs, c, h, w, z)

        return hidden_state, references, None
