# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from pytest_mock import MockerFixture

from nndet.nn.transformer.layers.conditional_detr import (
    ConditionalDETRTransformerDecoder,
)
from nndet.nn.transformer.layers.detr import (
    DETRTransformerDecoder,
    DETRTransformerEncoder,
)
from nndet.nn.transformer.transformer import DETRTransformer

TEST_CASES_SHAPE = [
    (
        DETRTransformerEncoder(embed_dim=256),
        DETRTransformerDecoder(embed_dim=256),
        torch.ones((4, 256, 8, 8, 8)),  # features
        torch.ones((24, 256)),  # query embed
        torch.ones((4, 256, 8, 8, 8)),  # pos_embed
    ),
    (
        DETRTransformerEncoder(num_layers=0, embed_dim=128),
        DETRTransformerDecoder(num_layers=3, embed_dim=128),
        torch.ones((4, 128, 3, 3, 3)),  # features
        torch.ones((12, 128)),  # query embed
        torch.ones((4, 128, 3, 3, 3)),  # pos_embed
    ),
    (
        DETRTransformerEncoder(embed_dim=256),
        ConditionalDETRTransformerDecoder(embed_dim=256),
        torch.ones((4, 256, 8, 8, 8)),  # features
        torch.ones((24, 256)),  # query embed
        torch.ones((4, 256, 8, 8, 8)),  # pos_embed
    ),
]
"""    (
        DETRTransformerEncoder(num_layers=0, embed_dim=128),
        ConditionalDETRTransformerDecoder(num_layers=3, embed_dim=128),
        torch.ones((4, 128, 3, 3, 3)),  # features
        torch.ones((12, 128)),  # query embed
        torch.ones((4, 128, 3, 3, 3)),  # pos_embed
    ),
]"""


class TestDETRTransformer:
    def test_transformer_settings(self, mocker: MockerFixture):
        encoder = mocker.MagicMock()

    @pytest.mark.parametrize("encoder,decoder,input_tensor,query_embed,pos_embed", TEST_CASES_SHAPE)
    def test_transformer_check_output_shape(self, encoder, decoder, input_tensor, query_embed, pos_embed):
        transformer = DETRTransformer(encoder=encoder, decoder=decoder)
        num_decoder_layers = transformer.decoder.layer_sequence.num_layers
        out_sequence, reference, encoder_output = transformer([input_tensor], query_embed, [pos_embed])
        out_shape = torch.Size([num_decoder_layers, input_tensor.shape[0], query_embed.shape[0], query_embed.shape[1]])
        assert out_sequence.shape == out_shape
