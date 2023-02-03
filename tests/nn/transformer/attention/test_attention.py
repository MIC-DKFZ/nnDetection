import pytest

from nndet.nn.transformer.attention.attention import MultiheadAttention

TEST_SETTINGS = [
    (64, 4, 1, 2, False),
    (512, 8, 0.2, 0.1, False),
]


@pytest.fixture
def attention():
    return MultiheadAttention()


class TestMultiheadAttention:
    @pytest.mark.parametrize("embed_dim,num_heads,attn_drop_value,proj_drop_value,batch_first", TEST_SETTINGS)
    def test_settings(self, embed_dim, num_heads, attn_drop_value, proj_drop_value, batch_first):
        attention = MultiheadAttention(embed_dim, num_heads, attn_drop_value, proj_drop_value, batch_first)
        assert attention.embed_dim == attention.attn.embed_dim == embed_dim
        assert attention.num_heads == attention.attn.num_heads == num_heads
        assert attention.attn.dropout == attn_drop_value
        assert attention.proj_drop.p == proj_drop_value

    # def test_output_shape(self):
