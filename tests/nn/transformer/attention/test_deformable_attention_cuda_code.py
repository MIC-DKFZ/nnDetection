# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from torch.autograd import gradcheck

from nndet.nn.transformer.attention.multi_scale_deform_attn_3d import (
    MultiScaleDeformableAttention,
    MultiScaleDeformableAttnFunction,
    multi_scale_deformable_attn_3d_pytorch,
)

# Large
# N, M, C = 1, 36, 32
# Lq, L, P = 10800, 3 ,4
# shapes = torch.as_tensor([(16, 15, 39), (8, 8, 20), (4, 4, 10)], dtype=torch.long).cuda()


# Medium
# N, M, C = 1, 16, 16 # M6,26, C32
# Lq, L, P = 4860, 3 ,4
# shapes = torch.as_tensor([(8, 15, 39), (4, 4, 10), (2, 2, 5)], dtype=torch.long).cuda()


TEST_SETTINGS = [
    (
        3,
        4,
        16,
        5,
        2,
        4,  # bs, num_heads, embed_dim, num_queries, num_levels, num_points
        torch.as_tensor([(3, 6, 4), (2, 3, 2)], dtype=torch.long).cuda(),
    ),  # shapes
]


# Tiny
# N, M, C = 1,1,1    #samples N, attention heads M, channels C
# Lq, L, P = 1,1,1   #query numbers/features Lq , Layers L (for multi-shape), reference ponits P
# shapes = torch.as_tensor([(2,2,2)], dtype=torch.long).cuda() # DxHxW shapes. Why DHW not HWD: https://discuss.pytorch.org/t/why-use-dxhxw-for-3d-input-data-instead-of-hxwxd/104045


@pytest.mark.parametrize("bs, num_heads, embed_dim, num_queries, num_levels, num_points, shapes", TEST_SETTINGS)
@torch.no_grad()
def test_forward_equal_with_pytorch_double(bs, num_heads, embed_dim, num_queries, num_levels, num_points, shapes):
    level_start_index = torch.cat(
        (shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1])
    )  # start indices of input values in a linear array of all pixels
    S = sum([(D * H * W).item() for D, H, W in shapes])  # total voxel count S
    value = torch.rand(bs, S, num_heads, embed_dim).cuda() * 0.01
    sampling_locations = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 3).cuda()
    attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)  # normalize
    im2col_step = bs  # im2col needs to be divisible by bs
    output_pytorch = (
        multi_scale_deformable_attn_3d_pytorch(
            value.double(), shapes, sampling_locations.double(), attention_weights.double()
        )
        .detach()
        .cpu()
    )
    output_cuda = (
        MultiScaleDeformableAttnFunction.apply(
            value.double(),
            shapes,
            level_start_index,
            sampling_locations.double(),
            attention_weights.double(),
            im2col_step,
        )
        .detach()
        .cpu()
    )
    assert torch.allclose(output_cuda, output_pytorch)


@pytest.mark.parametrize("bs, num_heads, embed_dim, num_queries, num_levels, num_points, shapes", TEST_SETTINGS)
@torch.no_grad()
def test_forward_equal_with_pytorch_float(bs, num_heads, embed_dim, num_queries, num_levels, num_points, shapes):
    level_start_index = torch.cat(
        (shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1])
    )  # start indices of input values in a linear array of all pixels
    S = sum([(D * H * W).item() for D, H, W in shapes])  # total voxel count S
    value = torch.rand(bs, S, num_heads, embed_dim).cuda() * 0.01
    sampling_locations = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 3).cuda()
    attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)  # normalize
    im2col_step = bs  # im2col needs to be divisible by bs
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
    assert torch.allclose(output_cuda, output_pytorch)


@pytest.mark.parametrize("bs, num_heads, embed_dim, num_queries, num_levels, num_points, shapes", TEST_SETTINGS)
def test_gradient_numerical(
    bs,
    num_heads,
    embed_dim,
    num_queries,
    num_levels,
    num_points,
    shapes,
    grad_value=True,
    grad_sampling_loc=True,
    grad_attn_weight=True,
):
    level_start_index = torch.cat(
        (shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1])
    )  # start indices of input values in a linear array of all pixels
    S = sum([(D * H * W).item() for D, H, W in shapes])  # total voxel count S
    value = torch.rand(bs, S, num_heads, embed_dim).cuda() * 0.01
    sampling_locations = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 3).cuda()
    attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = bs
    func = MultiScaleDeformableAttnFunction.apply

    value.requires_grad = grad_value
    sampling_locations.requires_grad = grad_sampling_loc
    attention_weights.requires_grad = grad_attn_weight

    gradok = gradcheck(
        func,
        (
            value.double(),
            shapes,
            level_start_index,
            sampling_locations.double(),
            attention_weights.double(),
            im2col_step,
        ),
    )
    assert gradok
