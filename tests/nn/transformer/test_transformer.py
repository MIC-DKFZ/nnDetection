import pytest

from nndet.nn.transformer.attention.attention import MultiheadAttention


@pytest.fixture
def attention():
    return MultiheadAttention(
        embed_dim=128,
        num_heads=8,
        attn_drop_value=0.1,
        proj_drop_value=0.1,
        batch_first=False,
    )


@pytest.fixture
def base_transformer_layer():
    return BaseTransformerLayer()
