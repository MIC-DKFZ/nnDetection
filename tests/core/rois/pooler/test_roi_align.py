# import pytest
# import torch
# from torchvision.ops.roi_align import roi_align as tvision_roi_align
# from nndet.core.rois.roi_align import roi_align_3d
# from nndet.core.rois.roi_align import roi_align

# # def test_roi_align():
# #     boxes = torch.tensor([[0.0, 2.0, 2.0, 4.0, 4.0]])
# #     fmap = torch.zeros(1, 1, 5, 5)
# #     fmap[:, :, 2:5, 2:5] = 1

# #     pooled_fmap = tvision_roi_align(
# #         fmap,
# #         boxes,
# #         output_size=(3, 3),
# #         spatial_scale=1.0,
# #         # sampling_ratio=2,
# #     )

# #     print(pooled_fmap)


# @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda gpu available")
# @pytest.mark.skipif(
#     roi_align_3d is None, reason="nnDetection was not build with GPU support"
# )
# def test_roi_align_3d():
#     boxes = torch.tensor([[0.0, 2.0, 2.0, 4.0, 4.0, 2.0, 4.0]])

#     fmap = torch.zeros(1, 1, 5, 5, 5, requires_grad=True)
#     # fmap[:, :, 2:5, 2:5, 2:5] = 1

#     pooled_fmap = roi_align(
#         fmap.cuda(),
#         boxes.cuda(),
#         output_size=(3, 3, 3),
#         spatial_scale=1.0,
#         # sampling_ratio=2,
#     )

#     loss = pooled_fmap.mean()
#     loss.backward()


# # test_roi_align()
# test_roi_align_3d()
