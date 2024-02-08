# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple

import pytest
import torch
from torch.autograd import gradcheck

from nndet.nn.transformer.attention.multi_scale_deform_attn_3d import (
    MultiScaleDeformableAttnFunction,
    ms_deform_import,
    multi_scale_deformable_attn_3d_pytorch,
)

TEST_SETTINGS = [
    (
        2,  # batch size
        8,  # num heads
        32,  # embed dim
        24,  # num queries
        3,  # num levels
        4,  # num_points
        torch.as_tensor([[16, 16, 16], [8, 8, 8], [4, 4, 4]], dtype=torch.long).cuda(),  # shapes
    ),
    (
        2,  # batch size
        8,  # num heads
        32,  # embed dim
        24,  # num queries
        3,  # num levels
        4,  # num_points
        torch.as_tensor([[16, 16, 8], [8, 8, 4], [4, 4, 2]], dtype=torch.long).cuda(),  # shapes
    ),
    (
        2,  # batch size
        8,  # num heads
        32,  # embed dim
        24,  # num queries
        3,  # num levels
        4,  # num_points
        torch.as_tensor([[8, 16, 16], [4, 8, 8], [2, 4, 4]], dtype=torch.long).cuda(),  # shapes
    ),
    (
        2,  # batch size
        8,  # num heads
        32,  # embed dim
        24,  # num queries
        3,  # num levels
        8,  # num_points
        torch.as_tensor([[16, 16, 16], [8, 8, 8], [4, 4, 4]], dtype=torch.long).cuda(),  # shapes
    ),
]


TEST_SETTINGS_GRAD_NUM = [
    (
        3,  # batch size
        4,  # num heads
        16,  # embed dim
        5,  # num queries
        2,  # num levels
        4,  # num_points
        torch.as_tensor([(3, 6, 4), (2, 3, 2)], dtype=torch.long).cuda(),  # shapes
    ),
]


def mini_network_grads(
    operator,
    bs,
    num_heads,
    embed_dim,
    num_queries,
    num_levels,
    num_points,
    shapes,
    pass_subset: bool,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.random.manual_seed(seed)
    level_start_index = torch.cat(
        (shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1])
    )  # start indices of input values in a linear array of all pixels
    S = sum([(D * H * W).item() for D, H, W in shapes])  # total voxel count S
    value = (
        torch.rand(
            bs,
            S,
            num_heads,
            embed_dim,
            requires_grad=True,
            device=device,
            dtype=torch.float,
        )
        * 0.01
    )
    sampling_locations = torch.rand(
        bs,
        num_queries,
        num_heads,
        num_levels,
        num_points,
        3,
        requires_grad=True,
        device=device,
        dtype=torch.float,
    )
    attention_weights = (
        torch.rand(
            bs,
            num_queries,
            num_heads,
            num_levels,
            num_points,
            requires_grad=True,
            device=device,
            dtype=torch.float,
        )
        + 1e-5
    )
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)  # normalize
    im2col_step = bs  # im2col needs to be divisible by bs

    pre1 = torch.nn.Linear(embed_dim, embed_dim)
    pre1.to(device=device)
    pre2 = torch.nn.Linear(3, 3)
    pre2.to(device=device)
    pre3 = torch.nn.Linear(num_points, num_points)
    pre3.to(device=device)
    post1 = torch.nn.Linear(num_heads * embed_dim, embed_dim)
    post1.to(device=device)

    value_pre = pre1(value)
    sampling_locations_pre = pre2(sampling_locations)
    attention_weights_pre = pre3(attention_weights)
    if pass_subset:
        output = operator(
            value_pre,
            shapes,
            sampling_locations_pre,
            attention_weights_pre,
        )
    else:
        output = operator(
            value_pre,
            shapes,
            level_start_index,
            sampling_locations_pre,
            attention_weights_pre,
            im2col_step,
        )

    output_m2 = post1(output)
    loss = output_m2.mean()
    loss.backward()
    return pre1.weight.grad, pre2.weight.grad, pre3.weight.grad, post1.weight.grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda gpu available")
@pytest.mark.skipif(not ms_deform_import, reason="nnDetection was not build with GPU support")
@pytest.mark.parametrize(
    "bs, num_heads, embed_dim, num_queries, num_levels, num_points, shapes",
    TEST_SETTINGS,
)
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
            value.double(),
            shapes,
            sampling_locations.double(),
            attention_weights.double(),
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda gpu available")
@pytest.mark.skipif(not ms_deform_import, reason="nnDetection was not build with GPU support")
@pytest.mark.parametrize(
    "bs, num_heads, embed_dim, num_queries, num_levels, num_points, shapes",
    TEST_SETTINGS,
)
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
            value,
            shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        )
        .detach()
        .cpu()
    )
    assert torch.allclose(output_cuda, output_pytorch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda gpu available")
@pytest.mark.skipif(not ms_deform_import, reason="nnDetection was not build with GPU support")
@pytest.mark.parametrize(
    "bs, num_heads, embed_dim, num_queries, num_levels, num_points, shapes",
    TEST_SETTINGS,
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_backward_equal_with_pytorch_float(
    bs,
    num_heads,
    embed_dim,
    num_queries,
    num_levels,
    num_points,
    shapes,
    seed,
):
    pre1_grad_expected, pre2_grad_expected, pre3_grad_expected, post1_grad_expected = mini_network_grads(
        operator=multi_scale_deformable_attn_3d_pytorch,
        bs=bs,
        num_heads=num_heads,
        embed_dim=embed_dim,
        num_queries=num_queries,
        num_levels=num_levels,
        num_points=num_points,
        shapes=shapes,
        pass_subset=True,
        seed=42,
        device=torch.device("cuda"),
    )

    pre1_grad, pre2_grad, pre3_grad, post1_grad = mini_network_grads(
        operator=MultiScaleDeformableAttnFunction.apply,
        bs=bs,
        num_heads=num_heads,
        embed_dim=embed_dim,
        num_queries=num_queries,
        num_levels=num_levels,
        num_points=num_points,
        shapes=shapes,
        pass_subset=False,
        seed=42,
        device=torch.device("cuda"),
    )

    assert pre1_grad_expected is not None
    assert pre2_grad_expected is not None
    assert pre3_grad_expected is not None
    assert post1_grad_expected is not None
    assert torch.allclose(pre1_grad, pre1_grad_expected)
    assert torch.allclose(pre2_grad, pre2_grad_expected)
    assert torch.allclose(pre3_grad, pre3_grad_expected)
    assert torch.allclose(post1_grad, post1_grad_expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda gpu available")
@pytest.mark.skipif(not ms_deform_import, reason="nnDetection was not build with GPU support")
@pytest.mark.parametrize(
    "bs, num_heads, embed_dim, num_queries, num_levels, num_points, shapes", TEST_SETTINGS_GRAD_NUM
)
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
