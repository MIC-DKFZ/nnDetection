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


class AbstractTransformer(nn.Module):
    """
    Abstract Transformer Class for DETR like models
    """

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
            Tensor: refined feature sequence (output of the encoder)
                (bs, C, H, W, (Z))
            Optional(Tensor): References from the transformer decoder
                ((num_decoder_layers), bs, num_queries, dim)
        """
        raise NotImplementedError
