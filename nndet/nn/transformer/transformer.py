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
from nndet.utils.typing import LINEARSEQ


class DETRTransformer(AbstractTransformer):
    def __init__(
        self,
        encoder: TransformerLayerSequence,
        decoder: TransformerLayerSequence,
        encoder_classifier: Optional[LINEARSEQ] = None,
        encoder_regressor: Optional[LINEARSEQ] = None,
        two_stage: bool = False,
        do_weight_init: bool = True,
    ):
        """
        Transformer module for DETR.
        Args:
            encoder: Transformer encoder
            decoder: Transformer decoder
            encoder_classifier:
            encoder_regressor:
        """
        super().__init__()
        if two_stage:
            raise ValueError("Two stage for DETR and Conditional DETR is not yet supported")
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_classifier = encoder_classifier
        self.encoder_regressor = encoder_regressor
        self.embed_dim = self.encoder.embed_dim
        self.dim = decoder.dim
        if do_weight_init:
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
        dim = self.dim
        assert dim == features.dim() - 2  # subtract batch size and sequence length
        if dim == 2:
            bs, c, x, y = features.shape
        elif dim == 3:
            bs, c, x, y, z = features.shape
        else:
            raise ValueError(f"Number of feature dimensions {dim} not supported.")

        features = features.view(bs, c, -1).permute(2, 0, 1)  # [bs, c, dims] -> [mul(dims), bs, c]
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # [num_query, c] -> [num_query, bs, c]

        if mask is not None:
            assert len(mask) == 0
            mask = mask[0].view(bs, -1)  # [bs, dims] -> [bs, mul(dims)]

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
            memory = memory.permute(1, 2, 0).reshape(bs, c, x, y)
        else:
            memory = memory.permute(1, 2, 0).reshape(bs, c, x, y, z)

        return hidden_state, references, None
