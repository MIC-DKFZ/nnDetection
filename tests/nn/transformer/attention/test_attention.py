import pytest

from nndet.nn.transformer.attention.attention import MultiheadAttention

TEST_SETTINGS = [
    (
        64,
        4,
        1,
        1,
        False,
    ),
    (512, 8, 0.1, 0.1, False),
]


class TestMultiheadAttention:
    @pytest.mark.parametrize("embed_dim,num_heads,attn_drop_value,proj_drop_value,batch_first")
    def test_settings(self, embed_dim, num_heads, attn_drop_value, proj_drop_value, batch_first):
        attention = MultiheadAttention(embed_dim, num_heads, attn_drop_value, proj_drop_value, batch_first)
        assert attention.embed_dim == embed_dim
        assert attention.num_heads == num_heads
        assert attention.proj_drop.p == attn_drop_value
