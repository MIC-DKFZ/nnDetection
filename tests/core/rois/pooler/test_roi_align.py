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
    boxes = torch.tensor(
        [[0.0, 2.0, 1.0, 4.0, 3.0]]
    )  # torchvision uses different axes - boxes ordering
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
def test_roi_align_3d_smoke(
    sampling_ratio, aligned, spatial_scale, output_size, n_boxes
):
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
