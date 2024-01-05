# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

from nndet.nn.transformer.attention.multi_scale_deform_attn_3d import (
    MultiScaleDeformableAttention,
    MultiScaleDeformableAttnFunction,
    ms_deform_import,
    multi_scale_deformable_attn_3d_pytorch,
)

TEST_SETTINGS_SELF = [
    (3, 1, 1, 1),
]


def get_grid_points(x, y, z):
    x_lin = torch.linspace(0, 1, x)
    y_lin = torch.linspace(0, 1, y)
    z_lin = torch.linspace(0, 1, z)
    grid_x, grid_y, grid_z = torch.meshgrid(x_lin, y_lin, z_lin, indexing="ij")
    img = torch.stack([grid_x, grid_y, grid_z], dim=0).unsqueeze(0)
    return img


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda gpu available")
@pytest.mark.skipif(not ms_deform_import, reason="nnDetection was not build with GPU support")
@pytest.mark.parametrize("embed_dim,num_heads,num_levels,num_points", TEST_SETTINGS_SELF)
def test_deformable_attention_xyz_dimension(embed_dim, num_heads, num_levels, num_points):
    # test which dimension is which
    bs = 1
    num_queries = 3
    level_start_index = torch.zeros(1, dtype=torch.int64).cuda()
    im2col_step = 2
    # value is feature map
    x, y, z = 32, 32, 32
    img = get_grid_points(x, y, z)
    temp_value = img.flatten(2).permute(0, 2, 1).unsqueeze(2)
    value = temp_value.cuda().contiguous()  # unsqueeze for num_heads
    test_img = temp_value.squeeze(2).transpose(1, 2).reshape(bs, embed_dim, x, y, z)  # sometimes seems to go wrong
    assert torch.allclose(img, test_img)
    # shape is shape of each feature level
    shapes = torch.tensor([x, y, z]).unsqueeze(0).cuda()
    # sampling locations are the normalized coordinates to sample from, sample from x, y, z
    sampling_locations = torch.tensor(
        [[[[[[1, 0, 0]]]], [[[[0, 1, 0]]]], [[[[0, 0, 1]]]]]], dtype=torch.float32
    ).cuda()  # bs, num_queries, num_heads
    # attention weights are the weights for each sample
    attention_weights = torch.ones(bs, num_queries, num_heads, num_levels, num_points).cuda()
    output_pytorch = (
        multi_scale_deformable_attn_3d_pytorch(value, shapes, sampling_locations, attention_weights).detach().cpu()
    )
    output_cuda = (
        MultiScaleDeformableAttnFunction.apply(
            value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step
        )
        .detach()
        .cpu()
    )
    assert torch.allclose(output_pytorch, output_cuda)
    # dimensions seem to be inverted
    expected = torch.tensor([[[0, 0, 0.125], [0, 0.125, 0], [0.125, 0, 0]]])
    assert torch.allclose(output_cuda, expected)
    assert torch.allclose(output_pytorch, expected)
