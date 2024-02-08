# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

import pytest
import torch

from nndet.nn.transformer.attention.multi_scale_deform_attn_3d import (
    MultiScaleDeformableAttention,
)


class ConstantOutput(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_val: float,
        expected_val: Optional[float] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_val = output_val
        self.expected_val = expected_val

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        assert data.shape[-1] == self.in_channels

        if self.expected_val is not None:
            expected_tensor = torch.empty_like(data).fill_(self.expected_val)
            assert torch.allclose(data, expected_tensor)

        prev_shape = data.shape[:-1]
        return torch.empty(*prev_shape, self.out_channels).fill_(self.output_val)


class CustomMultiScaleDeformableAttention(MultiScaleDeformableAttention):
    def __init__(self, *args, batch_size: int, ref_points_passed: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # query_pos 0
        # query 1
        # value 2

        self.sampling_offsets = ConstantOutput(
            in_channels=self.sampling_offsets.in_features,
            out_channels=self.sampling_offsets.out_features,
            output_val=4.0,
            expected_val=1.0,
        )
        self.attention_weights = ConstantOutput(
            in_channels=self.attention_weights.in_features,
            out_channels=self.attention_weights.out_features,
            output_val=5.0,
            expected_val=1.0,
        )
        self.value_proj = ConstantOutput(
            in_channels=self.value_proj.in_features,
            out_channels=self.value_proj.out_features,
            output_val=6.0,
            expected_val=2.0,
        )
        self.output_proj = ConstantOutput(
            in_channels=self.output_proj.in_features,
            out_channels=self.output_proj.out_features,
            output_val=7.0,
            expected_val=3.0,
        )

        self.batch_size = batch_size
        self.ref_points_passed = ref_points_passed
        self.red_embed_dim = self.embed_dim // self.num_heads

    def apply_deformable_attention(
        self,
        value: torch.Tensor,
        value_spatial_shapes: torch.Tensor,
        value_level_start_index: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        assert value.shape[0] == self.batch_size
        assert value.shape[2] == self.num_heads
        assert value.shape[3] == self.red_embed_dim

        assert value_spatial_shapes.shape == (self.num_levels, self.dim)
        assert value_level_start_index.shape == (self.num_levels,)

        assert sampling_locations.shape[0] == self.batch_size
        assert sampling_locations.shape[2] == self.num_heads
        assert sampling_locations.shape[3] == self.num_levels
        assert sampling_locations.shape[4] == self.num_points
        assert sampling_locations.shape[5] == self.dim

        assert attention_weights.shape[0] == self.batch_size
        assert attention_weights.shape[2] == self.num_heads
        assert attention_weights.shape[3] == self.num_levels
        assert attention_weights.shape[4] == self.num_points

        # check values
        # value comes from value_proj
        expected_value = torch.empty_like(value).fill_(6.0)
        assert torch.allclose(value, expected_value)

        if not self.ref_points_passed:
            # sampling_locations
            # centers (0) + offsets (4) / num_point * sizes (0) * 0.5 = 0
            expected_sampling_locations = torch.zeros_like(sampling_locations)
            assert torch.allclose(sampling_locations, expected_sampling_locations)
        else:
            # sampling_locations
            # centers (0) + offsets (4) / level_shapes
            pass

        # attention weights
        # weights.softmax(-1)
        attention_weights_expected = torch.empty_like(attention_weights).fill_(5.0).flatten(-2).softmax(-1)
        attention_weights_expected = attention_weights_expected.view_as(attention_weights)
        assert torch.allclose(attention_weights, attention_weights_expected)

        return torch.empty(self.batch_size, sampling_locations.shape[1], self.embed_dim).fill_(3.0)


@pytest.mark.parametrize("ref_points_passed", [True, False])
@pytest.mark.parametrize("batch_fist", [True, False])
def test_multi_scale_deformable_attention(batch_fist: bool, ref_points_passed: bool):
    custom_module = CustomMultiScaleDeformableAttention(
        embed_dim=32,
        num_heads=2,
        num_levels=3,
        num_points=4,
        img2col_step=2,
        batch_size=2,
        dropout=0.0,
        batch_first=batch_fist,
        ref_points_passed=ref_points_passed,
    )
    spatial_shapes = torch.tensor([[8, 8, 8], [4, 4, 4], [2, 2, 2]])
    level_start_index = torch.tensor([0, 8**3, 8**3 + 4**3])

    query = torch.ones(2, 12, 32)
    value = torch.empty(2, 8**3 + 4**3 + 2**3, 32).fill_(2.0)
    query_pos = torch.zeros(2, 12, 32)
    identity = torch.zeros_like(query)

    if ref_points_passed:
        refs_cccddd_norm = torch.zeros(2, 12, 3, 3)
    else:
        refs_cccddd_norm = torch.zeros(2, 12, 3, 6)

    if not batch_fist:
        query = query.permute(1, 0, 2)
        value = value.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        identity = identity.permute(1, 0, 2)

    output = custom_module(
        query=query,
        value=value,
        identity=identity,
        query_pos=query_pos,
        refs_cccddd_norm=refs_cccddd_norm,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )

    if not batch_fist:
        output = output.permute(1, 0, 2)

    expected_output = torch.empty(2, 12, 32).fill_(7.0)
    assert torch.allclose(output, expected_output)
