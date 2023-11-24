# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

# Parts of this code are from torchvision (https://github.com/pytorch/vision) licensed under
# SPDX-FileCopyrightText: 2016 Soumith Chintala
# SPDX-License-Identifier: BSD-3-Clause


import math
from typing import Callable, Sequence, Tuple, Union

import numpy as np
import pytest
import torch
from torchvision.ops.roi_align import roi_align as tvision_roi_align

from nndet.core.rois.pooler.roi_align import (  # RoIAlignBase,
    RoIAlignNaiveAssign,
    RoIAlignOrigAssign,
    roi_align,
    roi_align_3d,
)

###############################
# Test RoI Align Operation    #
###############################


def mini_network_roi_align(
    seed: int,
    roi_align_fn: Callable,
    roi_align_kwargs: dict,
    boxes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.random.manual_seed(seed)
    m1 = torch.nn.Conv3d(3, 3, 3, padding=1, bias=True)
    m1.to(device=boxes.device)
    m2 = torch.nn.Conv3d(3, 3, 3, padding=1, bias=True)
    m2.to(device=boxes.device)

    fmap = torch.rand(
        (2, 3, 16, 16, 16),
        requires_grad=True,
        device=boxes.device,
        dtype=boxes.dtype,
    )
    fmap_m1 = m1(fmap)
    pooled_fmap = roi_align_fn(
        fmap_m1,
        boxes,
        **roi_align_kwargs,
    )
    pooled_fmap_m2 = m2(pooled_fmap)
    loss = pooled_fmap_m2.mean()
    loss.backward()

    return m1.weight.grad, m2.weight.grad


# test with slow pytorch implementation
def trilinear_interpolation(data, x, y, z):
    """
    Adapted for 3D from source
    Source: https://github.com/pytorch/vision/blob/main/test/test_ops.py
    """
    # data is in x,y,z format
    size_x, size_y, size_z = data.shape

    if x <= 0:
        x = 0
    if y <= 0:
        y = 0
    if z <= 0:
        z = 0

    x_low = int(math.floor(x))
    y_low = int(math.floor(y))
    z_low = int(math.floor(z))

    if x_low >= size_x - 1:
        x_low = size_x - 1
        x_high = size_x - 1
        x = x_low
    else:
        x_high = x_low + 1

    if y_low >= size_y - 1:
        y_low = size_y - 1
        y_high = size_y - 1
        y = y_low
    else:
        y_high = y_low + 1

    if z_low >= size_z - 1:
        z_low = size_z - 1
        z_high = size_z - 1
        z = z_low
    else:
        z_high = z_low + 1

    dx_h = x - x_low
    dy_h = y - y_low
    dz_h = z - z_low
    dx_l = 1 - dx_h
    dy_l = 1 - dy_h
    dz_l = 1 - dz_h

    val = 0
    for dx, xp in zip((dx_l, dx_h), (x_low, x_high)):
        for dy, yp in zip((dy_l, dy_h), (y_low, y_high)):
            for dz, zp in zip((dz_l, dz_h), (z_low, z_high)):
                # if 0 <= xp < size_x and 0 <= yp < size_y and 0 <= zp < size_z:
                val += dx * dy * dz * data[xp, yp, zp]
    return val


def roi_align_3d_pytorch_slow(
    data,  # N, C, x, y, z
    boxes,  # R, 5 [batch_idx, x1, y1, x2, y2, z1, z2]
    output_size,  # sx, sy, sz
    spatial_scale: float = 1,
    sampling_ratio: int = -1,
    aligned: bool = False,
):
    """
    Adapted for 2D with nnDet format
    Source: https://github.com/pytorch/vision/blob/main/test/test_ops.py
    """
    dtype = data.dtype
    device = data.device

    n_boxes = boxes.size(0)
    n_b, n_channels, n_x, n_y, n_z = data.shape
    roi_size_x, roi_size_y, roi_size_z = output_size
    result = torch.zeros(
        n_boxes,
        n_channels,
        roi_size_x,
        roi_size_y,
        roi_size_z,
        dtype=dtype,
        device=device,
    )

    offset = 0.5 if aligned else 0.0
    if isinstance(spatial_scale, Sequence):
        assert len(spatial_scale) == 3
        _spatial_scale = [
            spatial_scale[0],
            spatial_scale[1],
            spatial_scale[0],
            spatial_scale[1],
            spatial_scale[2],
            spatial_scale[2],
        ]
    else:
        _spatial_scale = [spatial_scale] * 6

    for box_idx in range(n_boxes):
        box = boxes[box_idx]  # box [7] [batch_idx, x1, y1, x2, y2, z1, z2]
        batch_idx = int(box[0])

        bx_begin, by_begin, bx_end, by_end, bz_begin, bz_end = (
            x.item() * _spatial_scale[x_idx] - offset for x_idx, x in enumerate(box[1:])
        )

        box_size_x = bx_end - bx_begin
        box_size_y = by_end - by_begin
        box_size_z = bz_end - bz_begin
        bin_x = box_size_x / roi_size_x
        bin_y = box_size_y / roi_size_y
        bin_z = box_size_z / roi_size_z
        grid_x = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_x))
        grid_y = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_y))
        grid_z = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_z))

        for x in range(0, roi_size_x):
            start_x = bx_begin + x * bin_x
            for y in range(0, roi_size_y):
                start_y = by_begin + y * bin_y
                for z in range(0, roi_size_z):
                    start_z = bz_begin + z * bin_z
                    for channel in range(0, n_channels):
                        val = 0

                        for grid_x_idx in range(0, grid_x):
                            for grid_y_idx in range(0, grid_y):
                                for grid_z_idx in range(0, grid_z):
                                    point_x = start_x + (grid_x_idx + 0.5) * bin_x / grid_x
                                    point_y = start_y + (grid_y_idx + 0.5) * bin_y / grid_y
                                    point_z = start_z + (grid_z_idx + 0.5) * bin_z / grid_z

                                    val += trilinear_interpolation(
                                        data[batch_idx, channel],
                                        point_x,
                                        point_y,
                                        point_z,
                                    )
                        val /= grid_x * grid_y * grid_z
                        result[box_idx, channel, x, y, z] = val
    return result


# 32, 32, 32 feature map
BOXES = [
    [0.0, 2.0, 2.0, 5.0, 5.0, 2.0, 4.0],
    [0.0, 2.0, 8.0, 5.0, 12.0, 2.0, 4.0],
    [0.0, 0.0, 2.0, 7.0, 9.0, 2.0, 4.0],
    [0.0, 2.0, 2.0, 5.0, 5.0, 2.0, 4.0],
    [1.0, 2.0, 2.0, 5.0, 5.0, 5.0, 14.0],
    [1.0, 2.0, 2.0, 5.0, 5.0, 2.0, 4.0],
]
BOXES_BORDERS = [
    # [0.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
    [0.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    [0.0, 14.0, 14.0, 16.0, 16.0, 14.0, 16.0],
    [0.0, 14.0, 14.0, 15.0, 15.0, 14.0, 15.0],
    [0.0, 15.0, 15.0, 16.0, 16.0, 15.0, 16.0],
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda gpu available")
@pytest.mark.skipif(roi_align_3d is None, reason="nnDetection was not build with GPU support")
@pytest.mark.parametrize("aligned", [False])  # True, False])
@pytest.mark.parametrize(
    "spatial_scale",
    [
        1.0,
        # 0.5,
        # (0.5, 1.0, 1.0),
        # (1.0, 0.5, 1.0),
        # (1.0, 1.0, 0.5),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("output_size", [(3, 3, 3)])  # , (7, 7, 7)])
@pytest.mark.parametrize("sampling_ratio", [1])  # 0, 1, 2])
@pytest.mark.parametrize("seed", [0])  # , 1])
def test_roi_align_3d_vs_slow_pytorch_forward(
    seed: int,
    sampling_ratio: int,
    output_size: Tuple[int, int, int],
    device: str,
    dtype: torch.dtype,
    spatial_scale: Union[float, Tuple[float, float, float]],
    aligned: bool,
):
    tol = 1e-5
    if dtype == torch.float16:
        tol = 5e-3
    elif dtype == torch.bfloat16:
        tol = 5e-3

    boxes = torch.tensor(
        BOXES_BORDERS,
        requires_grad=False,
        dtype=dtype,
        device=device,
    )

    torch.random.manual_seed(seed)
    fmap = torch.rand(
        (2, 3, 16, 16, 16),
        requires_grad=False,
        device=device,
        dtype=dtype,
    )

    pooled_fmap = roi_align(
        fmap,
        boxes,
        output_size=output_size,
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio,
        aligned=aligned,
    )
    expected_pooled_fmap = roi_align_3d_pytorch_slow(
        fmap,
        boxes,
        output_size=output_size,
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio,
        aligned=aligned,
    )

    assert tuple(pooled_fmap.shape) == (boxes.size(0), 3, *output_size)
    assert torch.allclose(pooled_fmap, expected_pooled_fmap, rtol=tol, atol=tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda gpu available")
@pytest.mark.skipif(roi_align_3d is None, reason="nnDetection was not build with GPU support")
@pytest.mark.parametrize("aligned", [True])  # , False])
@pytest.mark.parametrize(
    "spatial_scale",
    [
        1.0,
        # 0.5,
        # (0.5, 1.0, 1.0),
        # (1.0, 0.5, 1.0),
        # (1.0, 1.0, 0.5),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("output_size", [(3, 3, 3)])  # , (7, 7, 7)])
@pytest.mark.parametrize("sampling_ratio", [1])  # 0, 1, 2])
@pytest.mark.parametrize("seed", [0])  # , 1])
def test_roi_align_3d_vs_slow_pytorch_backward(
    seed: int,
    sampling_ratio: int,
    output_size: Tuple[int, int, int],
    device: str,
    dtype: torch.dtype,
    spatial_scale: Union[float, Tuple[float, float, float]],
    aligned: bool,
):
    tol = 2e-5

    # input size 16, 16, 16
    boxes = torch.tensor(
        BOXES_BORDERS,
        requires_grad=False,
        dtype=dtype,
        device=device,
    )

    _seed = seed + 12345
    m1_grad, m2_grad = mini_network_roi_align(
        seed=_seed,
        roi_align_fn=roi_align,
        roi_align_kwargs={
            "output_size": output_size,
            "spatial_scale": spatial_scale,
            "sampling_ratio": sampling_ratio,
            "aligned": aligned,
        },
        boxes=boxes,
    )
    expected_m1_grad, expected_m2_grad = mini_network_roi_align(
        seed=_seed,
        roi_align_fn=roi_align_3d_pytorch_slow,
        roi_align_kwargs={
            "output_size": output_size,
            "spatial_scale": spatial_scale,
            "sampling_ratio": sampling_ratio,
            "aligned": aligned,
        },
        boxes=boxes,
    )
    print(m1_grad)
    print(expected_m1_grad)

    assert torch.allclose(m1_grad, expected_m1_grad, rtol=tol, atol=tol)
    assert torch.allclose(m2_grad, expected_m2_grad, rtol=tol, atol=tol)


# @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda gpu available")
# @pytest.mark.skipif(
#     roi_align_3d is None, reason="nnDetection was not build with GPU support"
# )
# @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
# @pytest.mark.parametrize("dim", [-1, -2, -3])
# @pytest.mark.parametrize("sampling_ratio", [0, 1, 2])
# @pytest.mark.parametrize("spatial_scale", [1.0, 2.0])
# def test_roi_align_pseudo2d_vs_tv(
#     seed: int,
#     dim: int,
#     sampling_ratio: int,
#     spatial_scale: float,
# ):
#     torch.manual_seed(seed)
#     boxes = torch.tensor(
#         [[0.0, 2.0, 1.0, 4.0, 3.0]]
#     )  # torchvision uses different axes - boxes ordering
#     fmap = torch.rand(1, 1, 10, 10, requires_grad=True)

#     pooled_fmap = tvision_roi_align(
#         fmap,
#         boxes,
#         output_size=(3, 3),
#         spatial_scale=spatial_scale,
#         sampling_ratio=sampling_ratio,
#     )
#     loss = torch.tensor(10) - pooled_fmap.sum()
#     loss.backward()

#     # 1.0, 2.0, 3.0, 4.0, 0.0, 1.0
#     s = 1.0 if int(spatial_scale) == 1 else 0.5
#     if dim == -1:
#         boxes_3d = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 0.0, s]])
#         output_size = (3, 3, 1)
#     elif dim == -2:
#         boxes_3d = torch.tensor([[0.0, 1.0, 0.0, 3.0, s, 2.0, 4.0]])
#         output_size = (3, 1, 3)
#     elif dim == -3:
#         boxes_3d = torch.tensor([[0.0, 0.0, 1.0, s, 3.0, 2.0, 4.0]])
#         output_size = (1, 3, 3)
#     else:
#         raise ValueError(f"Dim {dim} not supported in test")

#     boxes_3d = boxes_3d.cuda()
#     fmap_3d = fmap.detach().unsqueeze(dim=dim).cuda()
#     fmap_3d.requires_grad = True
#     pooled_fmap_3d = roi_align(
#         fmap_3d,
#         boxes_3d,
#         output_size=output_size,
#         spatial_scale=spatial_scale,
#         sampling_ratio=sampling_ratio,
#     )
#     loss3d = torch.tensor(10) - pooled_fmap_3d.sum()
#     loss3d.backward()

#     comp_pooled_fmap_3d = pooled_fmap_3d.squeeze(dim).cpu()
#     comp_fmap_3d_grad = fmap_3d.grad.squeeze(dim).cpu()

#     boxes_3d = boxes_3d.cpu()
#     fmap_3d = fmap_3d.cpu()
#     pooled_fmap_3d = pooled_fmap_3d.cpu()
#     loss3d = loss3d.cpu()

#     assert torch.allclose(pooled_fmap, comp_pooled_fmap_3d)
#     assert torch.allclose(loss, loss3d)
#     assert torch.allclose(fmap.grad, comp_fmap_3d_grad)
#     torch.cuda.empty_cache()


# @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda gpu available")
# @pytest.mark.skipif(
#     roi_align_3d is None, reason="nnDetection was not build with GPU support"
# )
# @pytest.mark.parametrize(
#     "proposal,match_expected,aligned",
#     [
#         ([[1.0, 1.0, 4.0, 4.0, 8.0, 8.0, 12.0]], True, False),
#         ([[1.0, 0.0, 4.0, 4.0, 8.0, 8.0, 12.0]], False, False),
#         ([[1.0, 1.0, 3.0, 4.0, 8.0, 8.0, 12.0]], False, False),
#         ([[1.0, 1.0, 4.0, 5.0, 8.0, 8.0, 12.0]], False, False),
#         ([[1.0, 1.0, 4.0, 4.0, 9.0, 8.0, 12.0]], False, False),
#         ([[1.0, 1.0, 4.0, 4.0, 8.0, 7.0, 12.0]], False, False),
#         ([[1.0, 1.0, 4.0, 4.0, 8.0, 8.0, 13.0]], False, False),
#         ([[1.0, 1.0, 4.0, 4.0, 8.0, 7.0, 13.0]], False, True),
#     ],
# )
# def test_roi_align_3d(proposal, match_expected, aligned):
#     boxes = torch.tensor(proposal)
#     fmap = torch.zeros(2, 1, 32, 32, 32)
#     fmap[1, :, 1:5, 4:9, 8:13] = 1

#     pooled_fmap = roi_align(
#         fmap.cuda(),
#         boxes.cuda(),
#         output_size=(3, 3, 3),
#         spatial_scale=1.0,
#         aligned=aligned,
#         sampling_ratio=1,
#     )

#     expected = torch.ones(1, 1, 3, 3, 3)
#     print(pooled_fmap)
#     if match_expected:
#         assert pooled_fmap.allclose(expected.cuda())
#     else:
#         assert not pooled_fmap.allclose(expected.cuda())


# @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda gpu available")
# @pytest.mark.skipif(
#     roi_align_3d is None, reason="nnDetection was not build with GPU support"
# )
# def test_roi_align_3d_scale_aniso():
#     boxes = torch.tensor([[1.0, 1.0, 4.0, 4.0, 8.0, 8.0, 12.0]])
#     fmap = torch.zeros(2, 1, 32, 32, 32)
#     fmap[1, :, 1:5, 4:9, 8:13] = 1

#     pooled_fmap = roi_align(
#         fmap.cuda(),
#         boxes.cuda(),
#         output_size=(3, 3, 3),
#         spatial_scale=1.0,
#         aligned=False,
#         sampling_ratio=1,
#     )

#     expected = torch.ones(1, 1, 3, 3, 3)
#     print(pooled_fmap)
#     assert pooled_fmap.allclose(expected.cuda())


# ############################
# # Test RoI Align Module    #
# ############################

# # def test_roi_align_base_masks():
# #     pass


# @pytest.fixture
# def pooler_naive():
#     return RoIAlignNaiveAssign((3, 3, 3))


# @pytest.fixture
# def pooler_orig():
#     return RoIAlignOrigAssign((3, 3, 3))


# def test_roi_align_naive_assign(pooler_naive):
#     boxes = torch.tensor(
#         [
#             [0, 0, 2, 2, 0, 2],
#             [0, 0, 4, 4, 0, 4],
#             [0, 0, 8, 8, 0, 8],
#             [0, 0, 16, 16, 0, 16],
#             [0, 0, 32, 32, 0, 32],
#             [0, 0, 64, 64, 0, 64],
#             [0, 0, 96, 96, 0, 96],
#             [0, 0, 128, 128, 0, 128],
#         ]
#     )

#     features = [0, 1, 2, 3]
#     image_size = (128, 128, 128)
#     levels = pooler_naive._find_pyramid_level(
#         boxes,
#         features,
#         image_size,
#     )
#     assert levels.allclose(torch.tensor([0, 0, 0, 1, 2, 3, 3, 4]))


# def test_roi_align_orig_assign(pooler_orig):
#     boxes = torch.tensor(
#         [
#             [0, 0, 2, 2, 0, 2],
#             [0, 0, 4, 4, 0, 4],
#             [0, 0, 8, 8, 0, 8],
#             [0, 0, 16, 16, 0, 16],
#             [0, 0, 32, 32, 0, 32],
#             [0, 0, 64, 64, 0, 64],
#             [0, 0, 96, 96, 0, 96],
#             [0, 0, 128, 128, 0, 128],
#         ]
#     )

#     features = [0, 1, 2, 3]
#     image_size = (128, 128, 128)
#     levels = pooler_orig._find_pyramid_level(
#         boxes,
#         features,
#         image_size,
#     )
#     assert levels.allclose(torch.tensor([0, 0, 0, 1, 2, 3, 4, 4]))
