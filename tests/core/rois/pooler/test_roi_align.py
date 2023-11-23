# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

# Parts of this code are from torchvision (https://github.com/pytorch/vision) licensed under
# SPDX-FileCopyrightText: 2016 Soumith Chintala
# SPDX-License-Identifier: BSD-3-Clause


import math
from typing import Tuple

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


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="No cuda gpu available",
)
@pytest.mark.skipif(
    roi_align_3d is None,
    reason="nnDetection was not build with GPU support",
)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("dim", [-1, -2, -3])
@pytest.mark.parametrize("sampling_ratio", [0, 1, 2])
@pytest.mark.parametrize("spatial_scale", [1.0, 2.0])
def test_roi_align_pseudo2d_vs_tv(
    seed: int,
    dim: int,
    sampling_ratio: int,
    spatial_scale: float,
):
    torch.manual_seed(seed)
    boxes = torch.tensor([[0.0, 2.0, 1.0, 4.0, 3.0]])  # torchvision uses different axes - boxes ordering
    fmap = torch.rand(1, 1, 10, 10, requires_grad=True)

    pooled_fmap = tvision_roi_align(
        fmap,
        boxes,
        output_size=(3, 3),
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio,
    )
    loss = torch.tensor(10) - pooled_fmap.sum()
    loss.backward()

    # 1.0, 2.0, 3.0, 4.0, 0.0, 1.0
    s = 1.0 if int(spatial_scale) == 1 else 0.5
    if dim == -1:
        boxes_3d = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 0.0, s]])
        output_size = (3, 3, 1)
    elif dim == -2:
        boxes_3d = torch.tensor([[0.0, 1.0, 0.0, 3.0, s, 2.0, 4.0]])
        output_size = (3, 1, 3)
    elif dim == -3:
        boxes_3d = torch.tensor([[0.0, 0.0, 1.0, s, 3.0, 2.0, 4.0]])
        output_size = (1, 3, 3)
    else:
        raise ValueError(f"Dim {dim} not supported in test")

    boxes_3d = boxes_3d.cuda()
    fmap_3d = fmap.detach().unsqueeze(dim=dim).cuda()
    fmap_3d.requires_grad = True
    pooled_fmap_3d = roi_align(
        fmap_3d,
        boxes_3d,
        output_size=output_size,
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio,
    )
    loss3d = torch.tensor(10) - pooled_fmap_3d.sum()
    loss3d.backward()

    comp_pooled_fmap_3d = pooled_fmap_3d.squeeze(dim).cpu()
    comp_fmap_3d_grad = fmap_3d.grad.squeeze(dim).cpu()

    boxes_3d = boxes_3d.cpu()
    fmap_3d = fmap_3d.cpu()
    pooled_fmap_3d = pooled_fmap_3d.cpu()
    loss3d = loss3d.cpu()

    assert torch.allclose(pooled_fmap, comp_pooled_fmap_3d)
    assert torch.allclose(loss, loss3d)
    assert torch.allclose(fmap.grad, comp_fmap_3d_grad)
    torch.cuda.empty_cache()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="No cuda gpu available",
)
@pytest.mark.skipif(
    roi_align_3d is None,
    reason="nnDetection was not build with GPU support",
)
@pytest.mark.parametrize("sampling_ratio", [-1, 0, 2, 4, 8])
@pytest.mark.parametrize("aligned", [True, False])
@pytest.mark.parametrize("spatial_scale", [1.0, 3.0, (1.0, 1.0, 1.0), (3.0, 3.0, 3.0)])
@pytest.mark.parametrize("output_size", [(3, 3, 3), (7, 7, 7)])
@pytest.mark.parametrize("n_boxes", [1, 3])
def test_roi_align_3d_smoke(sampling_ratio, aligned, spatial_scale, output_size, n_boxes):
    boxes = torch.tensor([[0.0, 2.0, 2.0, 4.0, 4.0, 2.0, 4.0]] * n_boxes)
    fmap = torch.zeros(1, 1, 16, 16, 16, requires_grad=True)

    pooled_fmap = roi_align(
        fmap.cuda(),
        boxes.cuda(),
        output_size=output_size,
        spatial_scale=spatial_scale,
        aligned=aligned,
        sampling_ratio=sampling_ratio,
    )

    loss = pooled_fmap.mean()
    loss.backward()

    expected = torch.zeros(n_boxes, 1, *output_size)
    assert pooled_fmap.allclose(expected.cuda())


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="No cuda gpu available",
)
@pytest.mark.skipif(
    roi_align_3d is None,
    reason="nnDetection was not build with GPU support",
)
@pytest.mark.parametrize(
    "proposal,match_expected",
    [
        ([[1.0, 1.0, 4.0, 4.0, 8.0, 8.0, 12.0]], True),
        ([[1.0, 0.0, 4.0, 4.0, 8.0, 8.0, 12.0]], False),
        ([[1.0, 1.0, 3.0, 4.0, 8.0, 8.0, 12.0]], False),
        ([[1.0, 1.0, 4.0, 5.0, 8.0, 8.0, 12.0]], False),
        ([[1.0, 1.0, 4.0, 4.0, 9.0, 8.0, 12.0]], False),
        ([[1.0, 1.0, 4.0, 4.0, 8.0, 7.0, 12.0]], False),
        ([[1.0, 1.0, 4.0, 4.0, 8.0, 8.0, 13.0]], False),
        # ([[1.0, 1.0, 4.0, 4.0, 8.0, 7.0, 13.0]], False), # TODO check this; since last point is outside this shouldn't be one?
    ],
)
@pytest.mark.skipif(
    roi_align_3d is None,
    reason="nnDetection was not build with GPU support",
)
def test_roi_align_3d(proposal, match_expected):
    boxes = torch.tensor(proposal)
    fmap = torch.zeros(2, 1, 32, 32, 32)
    fmap[1, :, 1:5, 4:9, 8:13] = 1

    pooled_fmap = roi_align(
        fmap.cuda(),
        boxes.cuda(),
        output_size=(3, 3, 3),
        spatial_scale=1.0,
        aligned=False,
        sampling_ratio=1,
    )

    expected = torch.ones(1, 1, 3, 3, 3)
    print(pooled_fmap)
    if match_expected:
        assert pooled_fmap.allclose(expected.cuda())
    else:
        assert not pooled_fmap.allclose(expected.cuda())


# @pytest.mark.skipif(
#     not torch.cuda.is_available(),
#     reason="No cuda gpu available",
# )
# @pytest.mark.skipif(
#     roi_align_3d is None,
#     reason="nnDetection was not build with GPU support",
# )
# def test_roi_align_3d_1px():
#     boxes = torch.tensor([[1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0]])
#     fmap = torch.zeros(2, 1, 32, 32, 32)
#     fmap[1, :, 1, 2, 3] = 1

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


# # # TODO: -1 corner check
# # # TODO: what is the expected input / output ?


# @pytest.mark.skipif(
#     not torch.cuda.is_available(),
#     reason="No cuda gpu available",
# )
# @pytest.mark.skipif(
#     roi_align_3d is None,
#     reason="nnDetection was not build with GPU support",
# )
# @pytest.mark.parametrize(
#     "proposal,match_expected",
#     [
#         ([[1.0, 0.0, 4.0, 4.0, 8.0, 8.0, 12.0]], True),
#         ([[1.0, 0.0, 3.0, 4.0, 8.0, 8.0, 12.0]], False),
#         ([[1.0, 0.0, 4.0, 5.0, 8.0, 8.0, 12.0]], False),
#         ([[1.0, 0.0, 4.0, 4.0, 9.0, 8.0, 12.0]], False),
#         ([[1.0, 0.0, 4.0, 4.0, 8.0, 7.0, 12.0]], False),
#         # ([[1.0, 0.0, 4.0, 4.0, 8.0, 8.0, 13.0]], False),
#     ],
# )
# def test_roi_align_3d_scale_iso(proposal, match_expected):
#     boxes = torch.tensor(proposal)
#     fmap = torch.zeros(2, 1, 4, 4, 4)
#     fmap[1, :, 0:1, 1:2, 3:4] = 1

#     pooled_fmap = roi_align(
#         fmap.cuda(),
#         boxes.cuda(),
#         output_size=(3, 3, 3),
#         spatial_scale=1 / 4.0,
#         aligned=False,
#         sampling_ratio=1,
#     )

#     expected = torch.ones(1, 1, 3, 3, 3)
#     print(pooled_fmap)
#     if match_expected:
#         assert pooled_fmap.allclose(expected.cuda())
#     else:
#         assert not pooled_fmap.allclose(expected.cuda())


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="No cuda gpu available",
)
@pytest.mark.skipif(
    roi_align_3d is None,
    reason="nnDetection was not build with GPU support",
)
def test_roi_align_3d_scale_aniso():
    boxes = torch.tensor([[1.0, 1.0, 4.0, 4.0, 8.0, 8.0, 12.0]])
    fmap = torch.zeros(2, 1, 32, 32, 32)
    fmap[1, :, 1:5, 4:9, 8:13] = 1

    pooled_fmap = roi_align(
        fmap.cuda(),
        boxes.cuda(),
        output_size=(3, 3, 3),
        spatial_scale=1.0,
        aligned=False,
        sampling_ratio=1,
    )

    expected = torch.ones(1, 1, 3, 3, 3)
    print(pooled_fmap)
    assert pooled_fmap.allclose(expected.cuda())


# test by simplified pytorch implementation


def trilinear_interpolation(data, x, y, z, snap_border=False):
    """
    Adapted for 3D from source
    Source: https://github.com/pytorch/vision/blob/main/test/test_ops.py
    """
    # data is in x,y,z format
    size_x, size_y, size_z = data.shape

    if snap_border:
        if -1 < x <= 0:
            x = 0
        elif size_x - 1 <= x < size_x:
            x = size_x - 1

        if -1 < y <= 0:
            y = 0
        elif size_y - 1 <= y < size_y:
            y = size_y - 1

        if -1 < z <= 0:
            z = 0
        elif size_z - 1 <= z < size_z:
            z = size_z - 1

    x_low = int(math.floor(x))
    y_low = int(math.floor(y))
    z_low = int(math.floor(z))
    x_high = x_low + 1
    y_high = y_low + 1
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
                if 0 <= xp < size_x and 0 <= yp < size_y and 0 <= zp < size_z:
                    val += dx * dy * dz * data[xp, yp, zp]
    return val


def bilinear_interpolation(data, x, y, snap_border=False):
    """
    Adapted for 2D with nnDet format
    Source: https://github.com/pytorch/vision/blob/main/test/test_ops.py
    """
    # data is in x,y format
    size_x, size_y = data.shape

    if snap_border:
        if -1 < x <= 0:
            x = 0
        elif size_x - 1 <= x < size_x:
            x = size_x - 1

        if -1 < y <= 0:
            y = 0
        elif size_y - 1 <= y < size_y:
            y = size_y - 1

    x_low = int(math.floor(x))
    y_low = int(math.floor(y))
    x_high = x_low + 1
    y_high = y_low + 1

    dx_h = x - x_low
    dy_h = y - y_low
    dx_l = 1 - dx_h
    dy_l = 1 - dy_h

    val = 0
    for dx, xp in zip((dx_l, dx_h), (x_low, x_high)):
        for dy, yp in zip((dy_l, dy_h), (y_low, y_high)):
            if 0 <= xp < size_x and 0 <= yp < size_y:
                val += dx * dy * data[xp, yp]
    return val


def roi_align_2d_pytorch_slow(
    data,  # N, C, x, y
    boxes,  # R, 5 [batch_idx, x1, y1, x2, y2]
    output_size,  # sx, sy
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

    n_channels = data.size(1)
    roi_size_x, roi_size_y = output_size
    result = torch.zeros(boxes.size(0), n_channels, roi_size_x, roi_size_y, dtype=dtype, device=device)

    offset = 0.5 if aligned else 0.0

    for box_idx, box in enumerate(boxes):  # box [5] [batch_idx, x1, y1, x2, y2]
        batch_idx = int(box[0])

        bx_begin, by_begin, bx_end, by_end = (x.item() * spatial_scale - offset for x in box[1:])

        box_size_x = bx_end - bx_begin
        box_size_y = by_end - by_begin
        bin_x = box_size_x / roi_size_x
        bin_y = box_size_y / roi_size_y

        for x in range(0, roi_size_x):
            # iterate x_begin ... x_begin + (sx - 1) * bin_x = x_begin + (sx - 1) * size_x / sx
            start_x = bx_begin + x * bin_x
            grid_x = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_x))

            for y in range(0, roi_size_y):
                start_y = by_begin + y * bin_y
                grid_y = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_y))

                for channel in range(0, n_channels):
                    val = 0

                    for grid_x_idx in range(0, grid_x):
                        for grid_y_idx in range(0, grid_y):
                            point_x = start_x + (grid_x_idx + 0.5) * bin_x / grid_x
                            point_y = start_y + (grid_y_idx + 0.5) * bin_y / grid_y

                            val += bilinear_interpolation(
                                data[batch_idx, channel],
                                point_x,
                                point_y,
                                snap_border=True,
                            )
                    val /= grid_x * grid_y
                    result[box_idx, channel, x, y] = val
        return result


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
    n_channels = data.size(1)
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

    for box_idx in range(n_boxes):
        box = boxes[box_idx]  # box [7] [batch_idx, x1, y1, x2, y2, z1, z2]
        batch_idx = int(box[0])

        bx_begin, by_begin, bx_end, by_end, bz_begin, bz_end = (x.item() * spatial_scale - offset for x in box[1:])

        box_size_x = bx_end - bx_begin
        box_size_y = by_end - by_begin
        box_size_z = bz_end - bz_begin
        bin_x = box_size_x / roi_size_x
        bin_y = box_size_y / roi_size_y
        bin_z = box_size_z / roi_size_z

        for x in range(0, roi_size_x):
            start_x = bx_begin + x * bin_x
            grid_x = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_x))

            for y in range(0, roi_size_y):
                start_y = by_begin + y * bin_y
                grid_y = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_y))

                for z in range(0, roi_size_z):
                    start_z = bz_begin + z * bin_z
                    grid_z = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_z))

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
                                        snap_border=True,
                                    )
                        val /= grid_x * grid_y * grid_z
                        result[box_idx, channel, x, y, z] = val
    return result


# @pytest.mark.skipif(
#     roi_align_3d is None,
#     reason="nnDetection was not build with GPU support",
# )
# @pytest.mark.parametrize("aligned", [True, False])
# @pytest.mark.parametrize("spatial_scale", [1.0, 3.0, (1.0, 1.0, 1.0), (3.0, 3.0, 3.0)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("output_size", [(3, 3, 3), (7, 7, 7)])
@pytest.mark.parametrize("sampling_ratio", [0, 1, 2])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_roi_align_3d_vs_slow_pytorch_forward(
    seed: int,
    sampling_ratio: int,
    output_size: Tuple[int, int, int],
    device: str,
    dtype: torch.dtype,
):
    spatial_scale = 1.0
    aligned = False

    tol = 1e-5
    if dtype is torch.float16:
        tol = 5e-3
    elif dtype == torch.bfloat16:
        tol = 5e-3

    boxes = torch.tensor(
        [
            [0.0, 2.0, 2.0, 5.0, 5.0, 2.0, 4.0],
            [0.0, 2.0, 8.0, 5.0, 12.0, 2.0, 4.0],
            [0.0, 0.0, 2.0, 7.0, 9.0, 2.0, 4.0],
            [0.0, 2.0, 2.0, 5.0, 5.0, 2.0, 4.0],
            [1.0, 2.0, 2.0, 5.0, 5.0, 5.0, 14.0],
            [1.0, 2.0, 2.0, 5.0, 5.0, 2.0, 4.0],
        ],
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

    # # backward

    # loss3d = torch.tensor(10) - expected_pooled_fmap.sum()
    # loss3d.backward()

    # print("hi")

    # def func(tmp):
    #     return roi_align_3d_pytorch_slow(
    #         tmp,
    #         boxes.cuda(),
    #         output_size=output_size,
    #         spatial_scale=spatial_scale,
    #         sampling_ratio=sampling_ratio,
    #         aligned=aligned,
    #     )

    # torch.autograd.gradcheck(func, (fmap_test_grad,), atol=1e-05)


############################
# Test RoI Align Module    #
############################

# def test_roi_align_base_masks():
#     pass


@pytest.fixture
def pooler_naive():
    return RoIAlignNaiveAssign((3, 3, 3))


@pytest.fixture
def pooler_orig():
    return RoIAlignOrigAssign((3, 3, 3))


def test_roi_align_naive_assign(pooler_naive):
    boxes = torch.tensor(
        [
            [0, 0, 2, 2, 0, 2],
            [0, 0, 4, 4, 0, 4],
            [0, 0, 8, 8, 0, 8],
            [0, 0, 16, 16, 0, 16],
            [0, 0, 32, 32, 0, 32],
            [0, 0, 64, 64, 0, 64],
            [0, 0, 96, 96, 0, 96],
            [0, 0, 128, 128, 0, 128],
        ]
    )

    features = [0, 1, 2, 3]
    image_size = (128, 128, 128)
    levels = pooler_naive._find_pyramid_level(
        boxes,
        features,
        image_size,
    )
    assert levels.allclose(torch.tensor([0, 0, 0, 1, 2, 3, 3, 4]))


def test_roi_align_orig_assign(pooler_orig):
    boxes = torch.tensor(
        [
            [0, 0, 2, 2, 0, 2],
            [0, 0, 4, 4, 0, 4],
            [0, 0, 8, 8, 0, 8],
            [0, 0, 16, 16, 0, 16],
            [0, 0, 32, 32, 0, 32],
            [0, 0, 64, 64, 0, 64],
            [0, 0, 96, 96, 0, 96],
            [0, 0, 128, 128, 0, 128],
        ]
    )

    features = [0, 1, 2, 3]
    image_size = (128, 128, 128)
    levels = pooler_orig._find_pyramid_level(
        boxes,
        features,
        image_size,
    )
    assert levels.allclose(torch.tensor([0, 0, 0, 1, 2, 3, 4, 4]))
