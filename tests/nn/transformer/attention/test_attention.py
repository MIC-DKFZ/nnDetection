import pytest
import torch

from nndet.nn.transformer.attention.attention import MultiheadAttention

TEST_SETTINGS = [
    (64, 4, 1, 2, False),
    (512, 8, 0.2, 0.1, False),
]

TEST_SHAPE = [
    (
        MultiheadAttention(
            embed_dim=512,
            num_heads=8,
        ),
        torch.ones((100, 4, 512)),  # [N, bs, C]
        torch.ones((100, 4, 512)),
        torch.ones((100, 4, 512)),
        (100, 4, 512),
    ),
    (
        MultiheadAttention(
            embed_dim=256,
            num_heads=16,
        ),
        torch.ones((40, 8, 256)),  # use different sequence lengths
        torch.ones((80, 8, 256)),
        torch.ones((80, 8, 256)),
        (40, 8, 256),
    ),
]


@pytest.fixture
def attention():
    return MultiheadAttention(
        embed_dim=512,
        num_heads=8,
    )


class TestMultiheadAttention:
    @pytest.mark.parametrize("embed_dim,num_heads,attn_drop_value,proj_drop_value,batch_first", TEST_SETTINGS)
    def test_settings(self, embed_dim, num_heads, attn_drop_value, proj_drop_value, batch_first):
        attention = MultiheadAttention(embed_dim, num_heads, attn_drop_value, proj_drop_value, batch_first)
        assert attention.embed_dim == attention.attn.embed_dim == embed_dim
        assert attention.num_heads == attention.attn.num_heads == num_heads
        assert attention.attn.dropout == attn_drop_value
        assert attention.proj_drop.p == proj_drop_value

    @pytest.mark.parametrize("attention,query,key,value,expected_out_shape", TEST_SHAPE)
    def test_output_shape(self, attention, query, key, value, expected_out_shape):
        out = attention(query, key, value)
        assert tuple(out.shape) == expected_out_shape
