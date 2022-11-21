from typing import Dict, Sequence, Tuple, Type

import pytest
import torch

from nndet.nn.layers.pos_embed.base import BasePositionEmbedding
from nndet.nn.layers.pos_embed.learned import PositionEmbeddingLearned
from nndet.nn.layers.pos_embed.sine import PositionEmbeddingSine

TEST_SHAPES = [
    (1, 48, 64, 64),
    (4, 24, 64, 64),
    (1, 48, 32, 64),
    (1, 48, 64, 32),
    (1, 48, 64, 64, 64),
    (4, 48, 64, 64, 64),
    (4, 24, 64, 64, 64),
    (1, 48, 32, 64, 64),
    (1, 48, 64, 32, 64),
    (1, 48, 64, 64, 32),
]

EMBED_OPT = [
    (PositionEmbeddingSine, {}),
    (PositionEmbeddingSine, {"normalize": True}),
    (PositionEmbeddingLearned, {}),
]


@pytest.mark.parametrize("input_shape", TEST_SHAPES)
@pytest.mark.parametrize("embed_cls", EMBED_OPT)
def test_smoke_pos_embed(input_shape: Sequence[int], embed_opt: Tuple[Type[BasePositionEmbedding], Dict]):
    embed_cls, embed_kwargs = embed_opt

    inp = torch.zeros(input_shape, dtype=torch.float)
    pos_embedding = embed_cls(dim=len(input_shape) - 2, num_pos_feats=input_shape[1], **embed_kwargs)
    pos = pos_embedding(inp)

    assert tuple(inp.shape) == tuple(pos.shape)
    assert inp.device == pos.device
